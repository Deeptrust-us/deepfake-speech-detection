#!/usr/bin/env python3
"""
Plot training curves from log files.

Usage:
    python tools/plot_training_curves.py --results_dir "HM-Conformer/results/Multilingual Test 15k/HM-Conformer"
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)

def parse_loss_file(loss_file):
    """Parse Loss.txt file and return list of loss values."""
    losses = []
    with open(loss_file, 'r') as f:
        for line in f:
            # Format: "Loss: 9.947659492492676"
            match = re.search(r'Loss:\s*([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
    return losses

def parse_metric_file(metric_file, metric_name='EER'):
    """Parse EER.txt or BestEER.txt file and return epochs and values."""
    epochs = []
    values = []
    with open(metric_file, 'r') as f:
        for line in f:
            # Format: "[5] EER: 49.932885906040276" or "[5] BestEER: 49.932885906040276"
            match = re.search(r'\[(\d+)\]\s*' + metric_name + r':\s*([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                values.append(float(match.group(2)))
    return epochs, values

def parse_loss_files(results_dir):
    """Parse all Loss*.txt files."""
    losses_dict = {}
    results_path = Path(results_dir)
    
    # Main loss file
    loss_file = results_path / 'Loss.txt'
    if loss_file.exists():
        losses_dict['Total Loss'] = parse_loss_file(loss_file)
    
    # Individual loss files (Loss0.txt, Loss1.txt, etc.)
    for i in range(10):  # Check up to Loss9.txt
        loss_file = results_path / f'Loss{i}.txt'
        if loss_file.exists():
            losses_dict[f'Loss {i}'] = parse_loss_file(loss_file)
    
    return losses_dict

def plot_training_curves(results_dir, output_file=None):
    """Plot training curves from log files."""
    results_path = Path(results_dir)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # 1. Plot Training Loss
    ax1 = axes[0, 0]
    losses_dict = parse_loss_files(results_dir)
    
    if 'Total Loss' in losses_dict:
        losses = losses_dict['Total Loss']
        # Since losses are logged continuously, we can plot them as steps
        steps = np.arange(len(losses))
        ax1.plot(steps, losses, alpha=0.6, linewidth=0.5, label='Training Loss')
        # Add moving average for better visualization
        window = max(1, len(losses) // 100)
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving Avg (window={window})', color='red')
        ax1.set_xlabel('Log Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot individual losses if available
    if len(losses_dict) > 1:
        ax1b = axes[0, 1]
        for key, losses in losses_dict.items():
            if key != 'Total Loss':
                steps = np.arange(len(losses))
                ax1b.plot(steps, losses, alpha=0.6, label=key)
        ax1b.set_xlabel('Log Steps')
        ax1b.set_ylabel('Loss')
        ax1b.set_title('Individual Loss Components')
        ax1b.legend()
        ax1b.grid(True, alpha=0.3)
    else:
        axes[0, 1].axis('off')
    
    # 2. Plot EER over epochs
    ax2 = axes[1, 0]
    eer_file = results_path / 'EER.txt'
    if eer_file.exists():
        epochs, eers = parse_metric_file(eer_file, 'EER')
        ax2.plot(epochs, eers, 'o-', linewidth=2, markersize=6, label='EER')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('EER (%)')
        ax2.set_title('Equal Error Rate (EER)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower EER is better
    
    # 3. Plot Best EER over epochs
    ax3 = axes[1, 1]
    best_eer_file = results_path / 'BestEER.txt'
    if best_eer_file.exists():
        epochs, best_eers = parse_metric_file(best_eer_file, 'BestEER')
        ax3.plot(epochs, best_eers, 's-', linewidth=2, markersize=6, label='Best EER', color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Best EER (%)')
        ax3.set_title('Best Equal Error Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # Lower EER is better
        
        # Add annotation for best value
        if len(best_eers) > 0:
            min_idx = np.argmin(best_eers)
            min_epoch = epochs[min_idx]
            min_eer = best_eers[min_idx]
            ax3.annotate(f'Best: {min_eer:.2f}% @ epoch {min_epoch}',
                        xy=(min_epoch, min_eer),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = results_path / 'training_curves.png'
    else:
        output_file = Path(output_file)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")
    
    # Also save as PDF
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Training curves saved to: {pdf_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from log files')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory containing log files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for the plot (default: results_dir/training_curves.png)')
    
    args = parser.parse_args()
    
    plot_training_curves(args.results_dir, args.output)

if __name__ == '__main__':
    main()

