import matplotlib.pyplot as plt
import numpy as np

# Define the 16 states: 8 coarse-valid (xxx0000) + 8 coarse-invalid (xxx0001)
states = [
    '0000000', '0010000', '0100000', '0110000', '1000000', '1010000', '1100000', '1110000',  # Coarse-valid
    # Coarse-invalid
    '0000001', '0010001', '0100001', '0110001', '1000001', '1010001', '1100001', '1110001'
]
state_labels = [f'$|{s}\\rangle$' for s in states]

# Helper function to create a bar plot


def plot_amplitudes(amplitudes, title, filename, colors, neg_amplitudes=None):
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(states)), amplitudes,
                   color=colors, edgecolor='black')

    # Add amplitude labels above or below bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{height:.3f}' if height >= 0 else f'{-height:.3f}'
        y_pos = - \
            0.05 if neg_amplitudes and neg_amplitudes[i] < 0 else height + 0.02
        plt.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                 ha='center', va='top' if neg_amplitudes and neg_amplitudes[i] < 0 else 'bottom',
                 fontsize=8, rotation=0)

    # Set labels and title
    plt.xticks(range(len(states)), state_labels, rotation=90, fontsize=10)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.ylim(-0.5, 1.0)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='blue',
                      edgecolor='black', label='Unmarked States'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red',
                      edgecolor='black', label='Coarse-Valid States'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green',
                      edgecolor='black', label='Solution State (|1010000⟩)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray',
                      edgecolor='black', label='Coarse-Invalid States')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Stage 1: Initialization
amplitudes = [1/np.sqrt(128)] * 16  # ~0.088
colors = ['blue'] * 16
plot_amplitudes(
    amplitudes, 'Step 1: Initialization - Uniform Superposition', 'step1.png', colors)

# Stage 2: Coarse Oracle Application (after ~4 iterations)
# Coarse-valid ~0.35, coarse-invalid ~0.0
amplitudes = [0.35] * 8 + [0.001] * 8
colors = ['red'] * 8 + ['gray'] * 8
plot_amplitudes(
    amplitudes, 'Step 2: Coarse Oracle Application - After ~4 Iterations', 'step2.png', colors)

# Stage 3: Fine Oracle Application
amplitudes = [0.35] * 8 + [0.001] * 8
amplitudes[5] = -0.35  # |1010000⟩ phase inverted
neg_amplitudes = [0.35] * 8 + [0.001] * 8
neg_amplitudes[5] = -0.35
colors = ['red'] * 8 + ['gray'] * 8
colors[5] = 'green'  # Mark solution state
plot_amplitudes(amplitudes, 'Step 3: Fine Oracle Application - Phase Inversion of Solution',
                'step3.png', colors, neg_amplitudes)

# Stage 4: Partial Diffusion
amplitudes = [0.20] * 8 + [0.001] * 8
amplitudes[5] = 0.30  # |1010000⟩ amplified
colors = ['red'] * 8 + ['gray'] * 8
colors[5] = 'green'
plot_amplitudes(
    amplitudes, 'Step 4: Partial Diffusion - Amplification in Coarse Subspace', 'step4.png', colors)

# Stage 5: Iteration (after ~2 fine iterations)
amplitudes = [0.10] * 8 + [0.001] * 8
amplitudes[5] = 0.35  # |1010000⟩ ~0.95
colors = ['red'] * 8 + ['gray'] * 8
colors[5] = 'green'
plot_amplitudes(
    amplitudes, 'Step 5: Iteration - After ~2 Fine Iterations', 'step5.png', colors)

# Stage 6: Measurement
amplitudes = [0.10] * 8 + [0.001] * 8
amplitudes[5] = 0.35  # Same as step 5
colors = ['red'] * 8 + ['gray'] * 8
colors[5] = 'green'
plot_amplitudes(
    amplitudes, 'Step 6: Prior to Measurement - Final Amplitudes', 'step6.png', colors)
