# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### 1. Forward Diffraction Simulation

```python
import torch
from litho_engine import FraunhoferDiffraction
from litho_engine.diffraction import create_test_mask

# Create diffraction model
diffraction = FraunhoferDiffraction(
    wavelength=193.0,  # Deep UV lithography
    pixel_size=10.0,   # nm
    NA=0.6             # Numerical aperture
)

# Create a mask pattern
mask = create_test_mask(size=128, pattern_type='square')

# Simulate diffraction
intensity = diffraction(mask)
```

### 2. Inverse Mask Optimization

```python
from litho_engine import MaskOptimizer

# Define your target pattern
target = create_test_mask(size=128, pattern_type='circle')

# Create optimizer
optimizer = MaskOptimizer(
    diffraction_model=diffraction,
    mask_shape=(128, 128),
    learning_rate=0.05,
    regularization=0.01
)

# Optimize mask
results = optimizer.optimize(
    target=target,
    num_iterations=100,
    verbose=True
)

optimized_mask = results['mask']
```

## Running Examples

```bash
# Forward diffraction with various patterns
python examples/forward_diffraction.py

# Inverse lithography optimization
python examples/inverse_optimization.py
```

## Running Tests

```bash
pytest tests/test_litho_engine.py -v
```

## Key Features

- **Fully Differentiable**: Built on PyTorch for automatic differentiation
- **GPU Accelerated**: Supports CUDA for fast computation
- **Physics-Based**: Implements Fraunhofer diffraction with proper optical modeling
- **Flexible**: Multiple loss functions, regularization options, and optimization strategies

## Advanced Features

### Custom Target Patterns

```python
# Create your own target
target = torch.zeros((128, 128))
target[40:88, 40:88] = 1.0  # Square region
```

### Adaptive Optimization

```python
from litho_engine.optimizer import AdaptiveMaskOptimizer

optimizer = AdaptiveMaskOptimizer(
    diffraction_model=diffraction,
    mask_shape=(128, 128),
    use_scheduler=True  # Adaptive learning rate
)

results = optimizer.optimize(
    target=target,
    early_stopping_patience=30
)
```

### GPU Acceleration

```python
# Move to GPU
diffraction = diffraction.cuda()
mask = mask.cuda()
target = target.cuda()

# Everything else stays the same!
intensity = diffraction(mask)
```

## Typical Workflow

1. **Define optical system parameters** (wavelength, NA, pixel size)
2. **Create or load target pattern** (desired output)
3. **Initialize optimizer** with diffraction model
4. **Run optimization** to find mask
5. **Validate results** by forward simulation
6. **Refine if needed** by adjusting parameters

## Performance Tips

- Start with smaller mask sizes (32×32 or 64×64) for faster iteration
- Use higher learning rates (0.05-0.1) for faster convergence
- Increase regularization (0.01-0.1) for smoother masks
- Enable GPU for masks larger than 256×256

## Troubleshooting

**Issue**: Optimization not converging
- **Solution**: Increase learning rate or reduce regularization

**Issue**: Mask too noisy
- **Solution**: Increase regularization weight

**Issue**: Slow performance
- **Solution**: Enable GPU or reduce mask size

## Next Steps

- Experiment with different optical parameters
- Try complex target patterns
- Modify loss functions for specific requirements
- Integrate resist models for end-to-end simulation

## References

See main README.md for detailed documentation and physics background.
