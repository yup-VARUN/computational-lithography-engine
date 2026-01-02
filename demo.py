#!/usr/bin/env python
"""
Simple command-line demo of the computational lithography engine.
"""

import torch
import argparse
from litho_engine import FraunhoferDiffraction, MaskOptimizer
from litho_engine.diffraction import create_test_mask


def demo_forward(mask_type='square', size=128):
    """Demonstrate forward diffraction."""
    print("\n" + "="*60)
    print("FORWARD DIFFRACTION DEMO")
    print("="*60)
    
    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)
    mask = create_test_mask(size=size, pattern_type=mask_type)
    
    print(f"\nMask: {mask_type} ({size}×{size})")
    print(f"Mask coverage: {mask.mean():.2%}")
    
    with torch.no_grad():
        intensity = diffraction(mask).squeeze()
    
    print(f"Output intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
    print(f"Total intensity: {intensity.sum():.2f}")
    print("\n✓ Forward simulation complete!")


def demo_inverse(target_type='circle', size=64, iterations=50):
    """Demonstrate inverse optimization."""
    print("\n" + "="*60)
    print("INVERSE OPTIMIZATION DEMO")
    print("="*60)
    
    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)
    target = create_test_mask(size=size, pattern_type=target_type)
    
    print(f"\nTarget: {target_type} ({size}×{size})")
    print(f"Target coverage: {target.mean():.2%}")
    print(f"Iterations: {iterations}")
    print("\nOptimizing...")
    
    optimizer = MaskOptimizer(
        diffraction_model=diffraction,
        mask_shape=(size, size),
        learning_rate=0.05,
        regularization=0.01
    )
    
    results = optimizer.optimize(
        target=target,
        num_iterations=iterations,
        verbose=False
    )
    
    initial_loss = results['loss_history'][0]
    final_loss = results['final_loss']
    improvement = (1 - final_loss/initial_loss) * 100
    
    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print("\n✓ Optimization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Computational Lithography Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode forward --pattern square --size 128
  %(prog)s --mode inverse --target circle --size 64 --iterations 50
  %(prog)s --mode both
        """
    )
    
    parser.add_argument('--mode', choices=['forward', 'inverse', 'both'],
                       default='both', help='Demo mode (default: both)')
    parser.add_argument('--pattern', choices=['square', 'circle', 'lines'],
                       default='square', help='Mask pattern for forward demo')
    parser.add_argument('--target', choices=['square', 'circle', 'lines'],
                       default='circle', help='Target pattern for inverse demo')
    parser.add_argument('--size', type=int, default=64,
                       help='Mask size (default: 64)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Optimization iterations (default: 50)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("COMPUTATIONAL LITHOGRAPHY ENGINE")
    print("="*60)
    
    if args.mode in ['forward', 'both']:
        demo_forward(args.pattern, args.size)
    
    if args.mode in ['inverse', 'both']:
        demo_inverse(args.target, args.size, args.iterations)
    
    print("\n" + "="*60)
    print("Demo complete! Check examples/ for more detailed demos.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
