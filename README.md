# ScalableSoftmax

An unofficial PyTorch implementation of Scalable-Softmax (Ssmax) from the paper "[Scalable-Softmax Is Superior for Attention](https://arxiv.org/pdf/2501.19399)" (Nakanishi, 2025).

## Overview

ScalableSoftmax is a drop-in replacement for standard Softmax that helps prevent attention fading in transformers by incorporating input size scaling. This helps maintain focused attention distributions even with large input sizes.

## Installation

```bash
pip install scalable-softmax
```

## Usage

```python
import torch
from scalable_softmax import ScalableSoftmax

# Initialize with default parameters
ssmax = ScalableSoftmax()

# Or customize parameters
ssmax = ScalableSoftmax(
    s=0.43,  # scaling parameter
    learn_scaling=True,  # make scaling parameter learnable
    bias=False  # whether to use bias term
)

# Apply to input tensor
x = torch.randn(batch_size, sequence_length)
output = ssmax(x)
```

## Features

- Drop-in replacement for standard softmax
- Learnable scaling parameter
- Optional bias term
- Maintains focused attention with large inputs

## Citation

```bibtex
@article{nakanishi2025scalable,
  title={Scalable-Softmax Is Superior for Attention},
  author={Nakanishi, Ken M.},
  journal={arXiv preprint arXiv:2501.19399},
  year={2025}
}
```

## License

MIT License
