# Sliced-Wasserstein Loss for LLM Quantization

## Overview

This document describes the Sliced-Wasserstein (SW) distance loss function implementation for LLM quantization in LightCompress. The SW loss can be applied to various blockwise optimization-based quantization methods as an alternative or complementary loss to traditional MSE/L2 losses.

## What is Sliced-Wasserstein Distance?

Sliced-Wasserstein distance is an efficient approximation of the Wasserstein distance (also known as Earth Mover's Distance) between two probability distributions. Unlike MSE which measures pointwise differences, SW distance measures the distributional difference between full-precision and quantized outputs, potentially leading to better preservation of the model's overall behavior.

### Key Properties

- **Distribution-aware**: Captures structural differences beyond pointwise errors
- **Efficient**: Uses random projections to reduce computational cost
- **Differentiable**: Fully compatible with gradient-based optimization
- **Flexible**: Can be used standalone or combined with traditional losses

## Implementation Details

### Loss Function API

The `LossFunction` class in `llmc/compression/quantization/train_utils.py` now supports the following loss methods:

#### Pure Sliced-Wasserstein Loss

```python
loss_func = LossFunction(
    method='sliced_wasserstein',  # or 'sw' as shorthand
    sw_num_projections=16,        # Number of random projections
    sw_block_size=None            # Block size (None = token-level)
)
```

#### Hybrid Losses

**MSE + SW:**
```python
loss_func = LossFunction(
    method='mse_sw',
    sw_num_projections=16,
    hybrid_weights={'base': 1.0, 'sw': 0.1}
)
```

**L2 + SW:**
```python
loss_func = LossFunction(
    method='l2_sw',
    sw_num_projections=16,
    hybrid_weights={'base': 1.0, 'sw': 0.1}
)
```

### Parameters

- **`method`** (str): Loss method name
  - `'sliced_wasserstein'` or `'sw'`: Pure SW loss
  - `'mse_sw'`: MSE + SW hybrid
  - `'l2_sw'`: L2 + SW hybrid
  - `'mse'`, `'l2'`, `'dist'`, `'kl'`: Original loss methods

- **`sw_num_projections`** (int, default=16): Number of random projections for SW distance
  - More projections = better approximation but slower computation
  - Typical range: 8-64
  - Recommended: 16-32 for most cases

- **`sw_block_size`** (int or None, default=None): Block size for SW computation
  - `None`: Token-level computation (treats each token independently)
  - Integer: Block-level computation (treats consecutive tokens as blocks)
  - Must divide sequence length evenly if specified

- **`hybrid_weights`** (dict): Weights for hybrid loss
  - `'base'`: Weight for MSE/L2 component (default: 1.0)
  - `'sw'`: Weight for SW component (default: 0.1)

- **`reduction`** (str, default='mean'): Reduction method for standard losses

## Usage

### Configuration File Method (Recommended)

The loss function can be configured via YAML config files. Add the following to the `special` section of your quantization config:

#### Example 1: Pure Sliced-Wasserstein Loss

```yaml
quant:
    method: OmniQuant  # Or TesseraQ, NormTweaking, etc.
    special:
        loss_method: sliced_wasserstein
        loss_kwargs:
            sw_num_projections: 16
            sw_block_size: null
            reduction: mean
        # ... other parameters ...
```

#### Example 2: Hybrid MSE + SW Loss

```yaml
quant:
    method: OmniQuant
    special:
        loss_method: mse_sw
        loss_kwargs:
            sw_num_projections: 32
            sw_block_size: null
            reduction: mean
            hybrid_weights:
                base: 1.0
                sw: 0.1
        # ... other parameters ...
```

#### Example 3: Block-level SW Loss

```yaml
quant:
    method: TesseraQ
    special:
        loss_method: l2_sw
        loss_kwargs:
            sw_num_projections: 16
            sw_block_size: 8  # Treat 8 consecutive tokens as one block
            hybrid_weights:
                base: 1.0
                sw: 0.15
        # ... other parameters ...
```

### Programmatic Method

If you're directly modifying the quantization method code:

```python
from llmc.compression.quantization.train_utils import LossFunction

# In your add_quant_config() method
loss_method = self.quant_config['special'].get('loss_method', 'mse')
loss_kwargs = self.quant_config['special'].get('loss_kwargs', {})
self.loss_func = LossFunction(method=loss_method, **loss_kwargs)
```

## Supported Quantization Methods

The SW loss is currently integrated with the following quantization methods (via `LossFunction`):

✅ **OmniQuant** - Fully supported
✅ **TesseraQ** - Fully supported
✅ **NormTweaking** - Fully supported

The following methods can be extended to support SW loss by converting inline loss computation to use `LossFunction`:

⚠️ **AWQ** - Requires modification to use `LossFunction`
⚠️ **OSPlus** - Requires modification to use `LossFunction`
⚠️ **GPTQ** - Requires special handling due to per-weight optimization

## Example Config Files

Pre-configured example files are available in:

- `configs/quantization/methods/OmniQuant/omniq_w_only_sw.yml` - Pure SW loss
- `configs/quantization/methods/OmniQuant/omniq_w_only_mse_sw.yml` - Hybrid MSE+SW
- `configs/quantization/methods/Tesseraq/tesseraq_w_only_sw.yml` - Hybrid L2+SW

## Hyperparameter Tuning Guide

### Number of Projections (`sw_num_projections`)

- **8-16**: Fast, suitable for quick experiments
- **16-32**: Recommended for production use
- **32-64**: High accuracy, slower computation

### Hybrid Weights

For `mse_sw` or `l2_sw`, adjust the relative weights:

- **Equal weighting**: `{'base': 1.0, 'sw': 1.0}`
- **MSE-dominant**: `{'base': 1.0, 'sw': 0.1}` (default, recommended starting point)
- **SW-dominant**: `{'base': 0.1, 'sw': 1.0}`
- **Custom**: Tune based on validation perplexity

### Block Size (`sw_block_size`)

- **`null` (token-level)**: Default, works well for most cases
- **4-16**: Experimental, may capture longer-range dependencies
- Must divide sequence length evenly

## Performance Considerations

### Computational Cost

- **Pure MSE/L2**: Baseline
- **Pure SW**: ~2-5x slower than MSE (depends on `sw_num_projections`)
- **Hybrid**: Slightly slower than pure SW (both losses computed)

### Memory Usage

- SW requires sorting, which may use more memory for large tensors
- Impact is typically negligible compared to model size

### Recommendations

1. Start with hybrid loss (`mse_sw` or `l2_sw`) with default weights
2. Use 16-32 projections for most experiments
3. Keep `sw_block_size=null` unless you have specific reasons
4. Monitor both training loss and validation perplexity

## Algorithm Details

### Token-Level SW Distance

For inputs of shape `[batch, seq_len, hidden_dim]`:

1. Flatten to `[N, D]` where `N = batch * seq_len`, `D = hidden_dim`
2. Sample `k` random unit vectors in `D`-dimensional space
3. Project both FP and quantized outputs onto these vectors
4. Sort projections and compute 1D Wasserstein distance for each
5. Average across all projections

### Block-Level SW Distance

For block size `b`:

1. Reshape to `[N, D]` where `N = batch * (seq_len/b)`, `D = b * hidden_dim`
2. Apply same algorithm as token-level

## Troubleshooting

### Error: "seq_len must be divisible by sw_block_size"

**Solution**: Set `sw_block_size: null` or choose a block size that divides your sequence length evenly.

### Error: "Unknown loss method"

**Solution**: Ensure you're using a supported loss method name. Check for typos in the config file.

### Training is too slow

**Solution**:
- Reduce `sw_num_projections` (try 8 or 16)
- Use hybrid loss with smaller SW weight
- Consider using SW only in later training epochs

### No improvement over MSE

**Solution**:
- Try different hybrid weight combinations
- Increase `sw_num_projections` for better approximation
- Experiment with block-level computation

## Citation

If you use the Sliced-Wasserstein loss in your research, please cite both LightCompress and relevant SW distance papers.

## Future Work

Potential extensions:

- [ ] Adaptive projection sampling
- [ ] Integration with AWQ, OSPlus, GPTQ
- [ ] Max-Sliced-Wasserstein variant
- [ ] Learnable projection vectors
- [ ] Multi-scale SW distance

## References

1. Original Sliced-Wasserstein Distance paper
2. Applications in neural network compression
3. LightCompress documentation
