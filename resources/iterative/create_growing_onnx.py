#!/usr/bin/env python3
"""
Script to create ONNX files with growing networks (increasing ReLU layers).
Each network follows the pattern: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
All intermediate dimensions are 4x4 for ReLU layers.
"""

import torch
import torch.nn as nn
import os

# Set random seed for reproducible weights
torch.manual_seed(42)

# Number word mappings for file names
NUM_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def get_num_word(n: int) -> str:
    """Convert number to word for file naming."""
    return NUM_WORDS.get(n, str(n))


def create_mlp_with_n_relus(n_relus: int, input_size: int = 4, hidden_size: int = 4, output_size: int = 4) -> nn.Sequential:
    """
    Create an MLP with n ReLU layers.

    Structure: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
    For n ReLUs, there are n+1 Linear layers.

    Args:
        n_relus: Number of ReLU layers
        input_size: Input dimension
        hidden_size: Hidden layer dimension (4 for all ReLU in/out)
        output_size: Output dimension

    Returns:
        nn.Sequential model
    """
    layers = []

    # First linear layer: input -> hidden
    layers.append(nn.Linear(input_size, hidden_size))

    # Add ReLU and Linear pairs
    for i in range(n_relus):
        layers.append(nn.ReLU())
        if i < n_relus - 1:
            # Intermediate: hidden -> hidden
            layers.append(nn.Linear(hidden_size, hidden_size))
        else:
            # Final: hidden -> output
            layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers)


def export_to_onnx(model: nn.Module, filepath: str, input_size: int = 4):
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )


def main():
    # Configuration - output to same directory as script
    output_dir = os.path.dirname(os.path.abspath(__file__))
    max_relus = 10  # Generate networks with 1 to 10 ReLU layers
    input_size = 4
    hidden_size = 4
    output_size = 4

    print(f"Creating growing ONNX networks in: {output_dir}")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    print("-" * 60)

    for n_relus in range(1, max_relus + 1):
        # Create model
        model = create_mlp_with_n_relus(n_relus, input_size, hidden_size, output_size)

        # Generate filename
        filename = f"{get_num_word(n_relus)}_mlp.onnx"
        filepath = os.path.join(output_dir, filename)

        # Export to ONNX
        export_to_onnx(model, filepath, input_size)

        # Print info
        n_linear = n_relus + 1
        print(f"Created {filename}: {n_relus} ReLU(s), {n_linear} Linear layers")

    print("-" * 60)
    print(f"Done! Created {max_relus} ONNX files.")


if __name__ == "__main__":
    main()
