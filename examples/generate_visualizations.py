#!/usr/bin/env python
"""
Generate Visualizations and Animations for Computational Lithography Engine
===========================================================================
All simulations use **EUV lithography** parameters (λ = 13.5 nm) with nm-scale
pixel sizes so that diffraction effects are clearly visible.

Produces:
  1. Animated GIFs of inverse mask optimization converging (multiple patterns)
  2. Static comparison of forward diffraction at multiple EUV NA settings
  3. Thermal compensation visualization (Si wafer cooling 200 °C → 80 °C)
  4. Arbitrary input shape gallery with ILT optimisation
  5. Multi-size (field-of-view) demonstration
  6. Thermal-aware ILT animation

All outputs saved to ``docs/images/`` for embedding in README.md.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio.v2 as imageio
import os
import sys
import io

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litho_engine import FraunhoferDiffraction, MaskOptimizer, ThermalExpansionModel
from litho_engine.optimizer import AdaptiveMaskOptimizer, ThermalAwareMaskOptimizer
from litho_engine.diffraction import create_test_mask

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images')
FINAL_FRAME_HOLD_COUNT = 15

# ── EUV defaults ──────────────────────────────────────────────────────────
EUV_WAVELENGTH = 13.5   # nm
EUV_NA = 0.33            # standard EUV NA
EUV_PIXEL = 1.0          # nm per pixel


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: add nm-scale axis labels to an Axes
# ---------------------------------------------------------------------------

def _set_nm_axes(ax, size, pixel_size, label=True):
    """Show tick marks in nanometres instead of pixel indices."""
    extent_nm = size * pixel_size
    ticks_px = np.linspace(0, size - 1, 5)
    labels_nm = [f"{t * pixel_size:.0f}" for t in ticks_px]
    ax.set_xticks(ticks_px)
    ax.set_xticklabels(labels_nm, fontsize=7)
    ax.set_yticks(ticks_px)
    ax.set_yticklabels(labels_nm, fontsize=7)
    if label:
        ax.set_xlabel('nm', fontsize=8)
        ax.set_ylabel('nm', fontsize=8)


# ---------------------------------------------------------------------------
# Arbitrary shape helpers
# ---------------------------------------------------------------------------

def create_arbitrary_pattern(size, pattern_name):
    """Create various arbitrary target patterns beyond the built-in ones."""
    mask = torch.zeros((size, size))
    center = size // 2

    if pattern_name == 'L_shape':
        w = size // 8
        h = size // 2
        mask[center - h:center + h, center - h:center - h + w] = 1.0
        mask[center + h - w:center + h, center - h:center + h // 2] = 1.0

    elif pattern_name == 'T_shape':
        w = size // 8
        h = size // 3
        mask[center - h // 2:center + h, center - w:center + w] = 1.0
        mask[center - h // 2 - w:center - h // 2 + w,
             center - h:center + h] = 1.0

    elif pattern_name == 'ring':
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist = torch.sqrt((x.float() - center) ** 2 + (y.float() - center) ** 2)
        r_outer = size // 4
        r_inner = size // 6
        mask[(dist <= r_outer) & (dist >= r_inner)] = 1.0

    elif pattern_name == 'cross':
        w = size // 10
        h = size // 3
        mask[center - w:center + w, center - h:center + h] = 1.0
        mask[center - h:center + h, center - w:center + w] = 1.0

    elif pattern_name == 'diamond':
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist = (torch.abs(x.float() - center) + torch.abs(y.float() - center))
        side = size // 4
        mask[dist <= side] = 1.0

    elif pattern_name == 'zigzag':
        w = size // 12
        seg = size // 6
        for i in range(4):
            y_start = center - size // 3 + i * seg
            y_end = y_start + seg
            if i % 2 == 0:
                x_start = center - size // 6
            else:
                x_start = center + size // 12
            mask[y_start:y_end, x_start:x_start + w] = 1.0
            if i < 3:
                x_lo = min(center - size // 6, center + size // 12)
                x_hi = max(center - size // 6 + w, center + size // 12 + w)
                mask[y_end - w:y_end, x_lo:x_hi] = 1.0

    else:
        mask = create_test_mask(size, pattern_type=pattern_name)

    return mask


# ---------------------------------------------------------------------------
# 1. Inverse mask optimization animation (GIF)
# ---------------------------------------------------------------------------

def generate_optimization_animation(pattern_name='cross', size=256,
                                    n_iterations=300, gif_name=None,
                                    wavelength=EUV_WAVELENGTH, NA=EUV_NA,
                                    pixel_size=EUV_PIXEL):
    """
    Animated GIF showing ILT converging.
    Frames: target | optimised mask | aerial image | loss curve.
    Axes annotated with nm-scale dimensions.
    """
    ensure_output_dir()
    if gif_name is None:
        gif_name = f'optimization_{pattern_name}.gif'

    fov_nm = size * pixel_size
    print(f"  Generating optimisation animation: {pattern_name}  "
          f"(λ={wavelength} nm, NA={NA}, pixel={pixel_size} nm, "
          f"FoV={fov_nm:.0f} nm) …")

    diffraction = FraunhoferDiffraction(
        wavelength=wavelength, pixel_size=pixel_size, NA=NA)
    target = create_arbitrary_pattern(size, pattern_name)

    mask_param = nn.Parameter(torch.rand(size, size))
    optimizer = optim.Adam([mask_param], lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25)

    frames = []
    loss_history = []
    capture_every = max(1, n_iterations // 60)

    for it in range(n_iterations):
        optimizer.zero_grad()
        mask_c = torch.sigmoid(mask_param)
        predicted = diffraction(mask_c)
        target_4d = target.unsqueeze(0).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(predicted, target_4d)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        loss_history.append(loss.item())

        if it % capture_every == 0 or it == n_iterations - 1:
            with torch.no_grad():
                pred_np = predicted.squeeze().cpu().numpy()
                mask_np = mask_c.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

            axes[0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
            axes[0].set_title('Target Pattern', fontsize=11, fontweight='bold')
            _set_nm_axes(axes[0], size, pixel_size)

            axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Optimised Mask (iter {it})',
                              fontsize=11, fontweight='bold')
            _set_nm_axes(axes[1], size, pixel_size)

            axes[2].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
            axes[2].set_title('Aerial Image (on wafer)',
                              fontsize=11, fontweight='bold')
            _set_nm_axes(axes[2], size, pixel_size)

            axes[3].plot(loss_history, color='#2196F3', linewidth=2)
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('MSE Loss')
            axes[3].set_title('Convergence', fontsize=11, fontweight='bold')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim(0, n_iterations)

            fig.suptitle(
                f'EUV Inverse Lithography — {pattern_name}  '
                f'(λ = {wavelength} nm, NA = {NA}, '
                f'FoV = {fov_nm:.0f} nm)',
                fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.92])

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)
            buf.close()

    for _ in range(FINAL_FRAME_HOLD_COUNT):
        frames.append(frames[-1])

    gif_path = os.path.join(OUTPUT_DIR, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    print(f"    ✓ Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# 2. Forward diffraction comparison (static) — EUV at different NAs
# ---------------------------------------------------------------------------

def generate_forward_diffraction_image():
    """
    Grid showing forward diffraction for EUV at multiple NA settings.
    Pixel size chosen so that features are near the resolution limit.
    """
    ensure_output_dir()
    print("  Generating forward diffraction comparison (EUV) …")

    patterns = ['square', 'circle', 'lines']
    configs = [
        {'wavelength': 13.5, 'NA': 0.25, 'pixel_size': 1.0,
         'label': 'EUV 13.5 nm, NA = 0.25'},
        {'wavelength': 13.5, 'NA': 0.33, 'pixel_size': 1.0,
         'label': 'EUV 13.5 nm, NA = 0.33'},
        {'wavelength': 13.5, 'NA': 0.55, 'pixel_size': 1.0,
         'label': 'EUV 13.5 nm, NA = 0.55 (High-NA)'},
    ]
    size = 256

    fig, axes = plt.subplots(len(configs), len(patterns) * 2,
                             figsize=(22, 11))

    for row, cfg in enumerate(configs):
        diffraction = FraunhoferDiffraction(
            wavelength=cfg['wavelength'], pixel_size=cfg['pixel_size'],
            NA=cfg['NA'])
        for col, pat in enumerate(patterns):
            mask = create_test_mask(size, pat)
            with torch.no_grad():
                intensity = diffraction(mask).squeeze().cpu().numpy()

            ax_mask = axes[row, col * 2]
            ax_int = axes[row, col * 2 + 1]

            ax_mask.imshow(mask.numpy(), cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(f'{pat} mask', fontsize=9)
            _set_nm_axes(ax_mask, size, cfg['pixel_size'], label=False)

            ax_int.imshow(intensity, cmap='inferno')
            ax_int.set_title('Aerial image', fontsize=9)
            _set_nm_axes(ax_int, size, cfg['pixel_size'], label=False)

        axes[row, 0].set_ylabel(cfg['label'], fontsize=10,
                                fontweight='bold', rotation=90, labelpad=15)

    fig.suptitle(
        'EUV Forward Diffraction — Varying Numerical Aperture\n'
        f'λ = 13.5 nm · pixel = 1 nm · field = {size} nm × {size} nm',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.06, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, 'forward_diffraction_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Thermal compensation visualization
# ---------------------------------------------------------------------------

def generate_thermal_compensation_image():
    """
    Show effect of thermal expansion compensation:
      Row 1 — without compensation → cooled pattern shrinks
      Row 2 — with compensation → cooled pattern matches target
    """
    ensure_output_dir()
    print("  Generating thermal compensation visualisation …")

    size = 256
    pixel_size = EUV_PIXEL
    target = create_arbitrary_pattern(size, 'cross')
    thermal = ThermalExpansionModel(process_temp=200.0, operating_temp=80.0)
    info = thermal.get_info()

    diffraction = FraunhoferDiffraction(
        wavelength=EUV_WAVELENGTH, pixel_size=pixel_size, NA=EUV_NA)

    # WITHOUT thermal compensation
    opt_no_comp = AdaptiveMaskOptimizer(
        diffraction, (size, size), learning_rate=0.05,
        regularization=0.001, use_scheduler=True)
    res_no_comp = opt_no_comp.optimize(
        target, num_iterations=300, verbose=False, early_stopping_patience=50)
    with torch.no_grad():
        aerial_no_comp = diffraction(res_no_comp['mask']).squeeze()
        cooled_no_comp = thermal.apply_thermal_contraction(aerial_no_comp)

    # WITH thermal compensation
    opt_comp = ThermalAwareMaskOptimizer(
        diffraction, (size, size), thermal,
        learning_rate=0.05, regularization=0.001, use_scheduler=True)
    res_comp = opt_comp.optimize(
        target, num_iterations=300, verbose=False, early_stopping_patience=50)
    with torch.no_grad():
        aerial_comp = diffraction(res_comp['mask']).squeeze()
        cooled_comp = thermal.apply_thermal_contraction(aerial_comp)

    diff_no_comp = torch.abs(cooled_no_comp - target)
    diff_comp = torch.abs(cooled_comp - target)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    titles_top = [
        f'Target (at {info["operating_temp_C"]}°C)',
        f'Aerial @ {info["process_temp_C"]}°C\n(no compensation)',
        f'Cooled to {info["operating_temp_C"]}°C\n(no compensation)',
        'Error (no comp.)'
    ]
    imgs_top = [target.numpy(), aerial_no_comp.numpy(),
                cooled_no_comp.numpy(), diff_no_comp.numpy()]
    titles_bot = [
        f'Target (at {info["operating_temp_C"]}°C)',
        f'Aerial @ {info["process_temp_C"]}°C\n(WITH compensation)',
        f'Cooled to {info["operating_temp_C"]}°C\n(WITH compensation)',
        'Error (WITH comp.)'
    ]
    imgs_bot = [target.numpy(), aerial_comp.numpy(),
                cooled_comp.numpy(), diff_comp.numpy()]

    for j in range(4):
        cmap = 'RdYlGn_r' if j == 3 else 'inferno'
        axes[0, j].imshow(imgs_top[j], cmap=cmap, vmin=0,
                          vmax=1 if j < 3 else None)
        axes[0, j].set_title(titles_top[j], fontsize=10, fontweight='bold')
        _set_nm_axes(axes[0, j], size, pixel_size, label=False)

        axes[1, j].imshow(imgs_bot[j], cmap=cmap, vmin=0,
                          vmax=1 if j < 3 else None)
        axes[1, j].set_title(titles_bot[j], fontsize=10, fontweight='bold')
        _set_nm_axes(axes[1, j], size, pixel_size, label=False)

    mse_no = torch.mean(diff_no_comp ** 2).item()
    mse_comp = torch.mean(diff_comp ** 2).item()
    fig.suptitle(
        f'Thermal Compensation — Si wafer {info["process_temp_C"]}°C → '
        f'{info["operating_temp_C"]}°C  '
        f'(contraction {info["contraction_ppm"]:.1f} ppm)\n'
        f'MSE w/o comp: {mse_no:.6f}  |  MSE w/ comp: {mse_comp:.6f}  '
        f'(EUV λ = {EUV_WAVELENGTH} nm, NA = {EUV_NA})',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    path = os.path.join(OUTPUT_DIR, 'thermal_compensation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Arbitrary shapes gallery
# ---------------------------------------------------------------------------

def generate_arbitrary_shapes_gallery():
    """
    ILT for several arbitrary input geometries under EUV.
    Clear diffraction visible in aerial images.
    """
    ensure_output_dir()
    print("  Generating arbitrary shapes gallery …")

    shapes = ['L_shape', 'T_shape', 'ring', 'cross', 'diamond', 'zigzag']
    size = 256
    pixel_size = EUV_PIXEL
    diffraction = FraunhoferDiffraction(
        wavelength=EUV_WAVELENGTH, pixel_size=pixel_size, NA=EUV_NA)

    fig, axes = plt.subplots(len(shapes), 3, figsize=(14, len(shapes) * 3.5))

    for i, name in enumerate(shapes):
        target = create_arbitrary_pattern(size, name)
        opt = AdaptiveMaskOptimizer(
            diffraction, (size, size), learning_rate=0.05,
            regularization=0.001, use_scheduler=True)
        res = opt.optimize(target, num_iterations=300, verbose=False,
                           early_stopping_patience=50)
        with torch.no_grad():
            pred = diffraction(res['mask']).squeeze().cpu().numpy()

        axes[i, 0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
        axes[i, 0].set_title('Target', fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 0], size, pixel_size, label=False)

        axes[i, 1].imshow(res['mask'].squeeze().cpu().numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Optimised Mask', fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 1], size, pixel_size, label=False)

        axes[i, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Aerial Image (loss={res["final_loss"]:.4f})',
                             fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 2], size, pixel_size, label=False)

        axes[i, 0].set_ylabel(name, fontsize=11, fontweight='bold',
                              rotation=0, labelpad=60, va='center')

    fig.suptitle(
        f'EUV Inverse Lithography — Arbitrary Input Shapes\n'
        f'λ = {EUV_WAVELENGTH} nm · NA = {EUV_NA} · '
        f'pixel = {pixel_size} nm · field = {size} nm × {size} nm',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, 'arbitrary_shapes_gallery.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Multi-size demonstration
# ---------------------------------------------------------------------------

def generate_multi_size_image():
    """Show the engine works at different field-of-view sizes (all EUV)."""
    ensure_output_dir()
    print("  Generating multi-size demonstration …")

    sizes = [128, 256, 512]
    pixel_size = EUV_PIXEL
    diffraction = FraunhoferDiffraction(
        wavelength=EUV_WAVELENGTH, pixel_size=pixel_size, NA=EUV_NA)

    fig, axes = plt.subplots(len(sizes), 3, figsize=(14, len(sizes) * 4))

    for i, sz in enumerate(sizes):
        target = create_arbitrary_pattern(sz, 'cross')
        iters = 200 if sz <= 256 else 150  # fewer iters for large grids to limit runtime
        opt = AdaptiveMaskOptimizer(
            diffraction, (sz, sz), learning_rate=0.05,
            regularization=0.001, use_scheduler=True)
        res = opt.optimize(target, num_iterations=iters, verbose=False,
                           early_stopping_patience=40)
        with torch.no_grad():
            pred = diffraction(res['mask']).squeeze().cpu().numpy()

        fov = sz * pixel_size
        axes[i, 0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
        axes[i, 0].set_title(
            f'Target ({sz}×{sz} px, {fov:.0f}×{fov:.0f} nm)',
            fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 0], sz, pixel_size, label=False)

        axes[i, 1].imshow(res['mask'].squeeze().cpu().numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Optimised Mask', fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 1], sz, pixel_size, label=False)

        axes[i, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Aerial Image (loss={res["final_loss"]:.4f})',
                             fontsize=10, fontweight='bold')
        _set_nm_axes(axes[i, 2], sz, pixel_size, label=False)

    fig.suptitle(
        f'EUV ILT — Multiple Field-of-View Sizes\n'
        f'λ = {EUV_WAVELENGTH} nm · NA = {EUV_NA} · pixel = {pixel_size} nm',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, 'multi_size_demo.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Thermal-aware optimisation animation
# ---------------------------------------------------------------------------

def generate_thermal_optimization_animation(size=256, n_iterations=300):
    """
    GIF: thermal-aware ILT — mask converges such that the cooled wafer
    pattern matches the target.
    """
    ensure_output_dir()
    print("  Generating thermal-aware optimisation animation …")

    pixel_size = EUV_PIXEL
    fov_nm = size * pixel_size
    target = create_arbitrary_pattern(size, 'diamond')
    thermal = ThermalExpansionModel(process_temp=200.0, operating_temp=80.0)
    diffraction = FraunhoferDiffraction(
        wavelength=EUV_WAVELENGTH, pixel_size=pixel_size, NA=EUV_NA)
    info = thermal.get_info()

    compensated_target = thermal.apply_thermal_precompensation(target)

    mask_param = nn.Parameter(torch.rand(size, size))
    optimizer = optim.Adam([mask_param], lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25)

    frames = []
    loss_history = []
    capture_every = max(1, n_iterations // 60)

    for it in range(n_iterations):
        optimizer.zero_grad()
        mask_c = torch.sigmoid(mask_param)
        predicted = diffraction(mask_c)
        target_4d = compensated_target.unsqueeze(0).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(predicted, target_4d)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        loss_history.append(loss.item())

        if it % capture_every == 0 or it == n_iterations - 1:
            with torch.no_grad():
                pred_np = predicted.squeeze().cpu().numpy()
                mask_np = mask_c.squeeze().cpu().numpy()
                cooled = thermal.apply_thermal_contraction(
                    predicted.squeeze()).cpu().numpy()

            fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))

            axes[0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
            axes[0].set_title(f'Target ({info["operating_temp_C"]}°C)',
                              fontsize=10, fontweight='bold')
            _set_nm_axes(axes[0], size, pixel_size)

            axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask (iter {it})',
                              fontsize=10, fontweight='bold')
            _set_nm_axes(axes[1], size, pixel_size)

            axes[2].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
            axes[2].set_title(f'Aerial @ {info["process_temp_C"]}°C',
                              fontsize=10, fontweight='bold')
            _set_nm_axes(axes[2], size, pixel_size)

            axes[3].imshow(cooled, cmap='inferno', vmin=0, vmax=1)
            axes[3].set_title(f'Cooled to {info["operating_temp_C"]}°C',
                              fontsize=10, fontweight='bold')
            _set_nm_axes(axes[3], size, pixel_size)

            axes[4].plot(loss_history, color='#FF5722', linewidth=2)
            axes[4].set_xlabel('Iteration')
            axes[4].set_ylabel('MSE Loss')
            axes[4].set_title('Convergence', fontsize=10, fontweight='bold')
            axes[4].set_yscale('log')
            axes[4].grid(True, alpha=0.3)
            axes[4].set_xlim(0, n_iterations)

            fig.suptitle(
                f'Thermal-Aware EUV ILT — '
                f'{info["process_temp_C"]}°C → {info["operating_temp_C"]}°C  '
                f'(contraction {info["contraction_ppm"]:.1f} ppm)  ·  '
                f'λ = {EUV_WAVELENGTH} nm, NA = {EUV_NA}, '
                f'FoV = {fov_nm:.0f} nm',
                fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.91])

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)
            buf.close()

    for _ in range(FINAL_FRAME_HOLD_COUNT):
        frames.append(frames[-1])

    gif_path = os.path.join(OUTPUT_DIR, 'thermal_optimization.gif')
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    print(f"    ✓ Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Generating EUV Lithography Visualisations & Animations")
    print(f"  λ = {EUV_WAVELENGTH} nm · NA = {EUV_NA} · "
          f"pixel = {EUV_PIXEL} nm")
    print("=" * 70)

    # 1. Optimisation animations for various patterns
    for pat in ['cross', 'ring', 'diamond']:
        generate_optimization_animation(pat, size=256, n_iterations=300)

    # 2. Forward diffraction comparison (multiple NAs)
    generate_forward_diffraction_image()

    # 3. Thermal compensation
    generate_thermal_compensation_image()

    # 4. Arbitrary shapes gallery
    generate_arbitrary_shapes_gallery()

    # 5. Multi-size demo
    generate_multi_size_image()

    # 6. Thermal-aware optimisation animation
    generate_thermal_optimization_animation()

    print("\n" + "=" * 70)
    print("  All visualisations generated successfully!")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
