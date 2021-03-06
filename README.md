## Wide Residual Networks

[![Travis](https://flat.badgen.net/travis/paradoxysm/wideresnet?label=build)](https://travis-ci.com/paradoxysm/wideresnet)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/wideresnet?label=coverage)](https://codecov.io/gh/paradoxysm/wideresnet)
[![GitHub](https://flat.badgen.net/github/license/paradoxysm/wideresnet)](https://github.com/paradoxysm/wideresnet/blob/master/LICENSE)

Keras (TensorFlow 2.4) and PyTorch (v1.8) implementations that are faithful to the original WideResNet proposed in [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146) and implemented in [here](https://github.com/szagoruyko/wide-residual-networks).

PyTorch implementation is nearly identical to the original, except without the use of Torchnet's Engine and is implemented within a class.

Keras/TensorFlow implementation is built similarly in a class, but requires a number of auxiliary changes to underlying TensorFlow functions to properly replicate the PyTorch implementation. To the best of my knowledge, this is the most accurate (faithful to the original implementation) publicly available re-implementation of the WideResNet in a framework other than torch.

### Replicating WideResNet: Implementation Details

In order for a Keras/TensorFlow implementation to accurately follow the original PyTorch implementation, the following needs to be considered:
 * ***Weight Initialization***: 
   * WideResNet uses a Kaiming/He Normal initialization; this is not default behaviour
   * TensorFlow's He Normal is a truncated form while Pytorch implements an untruncated form as Kaiming Normal
 * ***Bias Initialization***: WideResNet does not use biases in the convolutional layers; this is not default behaviour
 * ***Padding***: TensorFlow Convolution Layers do not have capacity for explicit padding; a ZeroPadding2D Layer must precede it
 * ***Batch Normalization***:
    * WideResNet initializes weights/gamma from a uniform distribution; this is not default behaviour
    * TensorFlow uses an epsilon of 1e-3 and a momentum of 0.99; PyTorch uses an epsilon of 1e-5 and a "momentum" of 0.1 (TensorFlow equivalent of 0.9)
 * ***Shuffling***: TensorFlow does not truly shuffle the same way as PyTorch unless the buffer is the size of the entire dataset
 * ***Weight Decay***: TensorFlow has no global weight decay like PyTorch; Implement L2 Regularizers for both kernels/weights and biases for ALL trainable layers (i.e. Conv2D, BatchNorm, Dense). Alternatively, implement a custom SGD optimizer.
 * ***SGD with Momentum***: PyTorch implements momentum differently from Sutskever et. al. and TensorFlow to the effect that the moving average of momentum is invariant to changes in learning rate.

## Installation

Once you have a suitable python environment setup, and this repository has been downloaded locally, `wideresnet` can be easily installed using `pip`:
```
pip install -r requirements.txt
```
> `wideresnet` is tested and supported on Python 3.6 up to Python 3.8. Usage on other versions of Python is not guaranteed to work as intended.

## Usage

### Keras

The Keras implementation works with TensorFlow Datasets.
```python
# Load the CIFAR10 Dataset
import tensorflow_datasets as tfds
train = tfds.load('cifar10', split='train', as_supervised=True)
test = tfds.load('cifar10', split='test', as_supervised=True)

# Create the model
from wideresnet.keras import WideResNet
from wideresnet import configs
model = WideResNet(**configs['cifar10'])

# Train and Validate the model
model.fit(train, val=test)
```

### PyTorch

The PyTorch implementation works with PyTorch's DataLoader.
```python
# Ready the CIFAR10 Dataset
import torchvision.datasets as tvds
data = tvds.CIFAR10

# Create the model
from wideresnet.pytorch import WideResNet
from wideresnet import configs
model = WideResNet(**configs['cifar10'])

# Train and Validate the model
model.fit(data, val=data)
```

## Changelog

See the [changelog](https://github.com/paradoxysm/wideresnet/blob/master/CHANGES.md) for a history of notable changes to `wideresnet`.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/wideresnet?style=flat-square)](https://codeclimate.com/github/paradoxysm/wideresnet/maintainability)

`wideresnet` has been worked on with the CIFAR10 dataset and is complete in this regard. It was not fully validated in it's reproducibility of other datasets compared to the original. Finally, the TensorFlow changes to the SGD optimizer only include within the scope of its use for WideResNet (i.e. weight decay and Nesterov momentum for dense tensors). Sparse tensors or non-momentum SGD are implemented in the original TensorFlow manner. Any help on rectifying that gap would be appreciated!

## Help and Support

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/wideresnet/issues).
