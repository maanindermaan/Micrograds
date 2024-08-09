# Micrograd

Micrograd is a small, educational autograd engine implemented in Python. This project is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and aims to provide a simple, understandable implementation of automatic differentiation and basic neural networks.

## Features

- Scalar-valued autograd engine
- Simple neural network library
- Supports basic arithmetic operations and common activation functions
- Implements backpropagation for automatic gradient computation

## Key Components

### Value Class

The `Value` class is the core of the autograd engine. It supports:

- Basic arithmetic operations (+, -, *, /)
- Power operation
- Exponential and hyperbolic tangent functions
- Automatic computation of gradients

### Neural Network Components

- `Neuron`: Represents a single neuron with weights and bias
- `Layer`: A collection of neurons
- `MLP` (Multi-Layer Perceptron): Composed of multiple layers

## Usage

### Basic Operations

```python
from micrograd import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + Value(10.0)
print(c.data)  # Output: 4.0

c.backward()
print(a.grad)  # Output: -3.0
print(b.grad)  # Output: 2.0
```

### Creating and Training a Neural Network

```python
from micrograd import MLP

# Create a neural network with 3 inputs, two hidden layers of 4 neurons each, and 1 output
n = MLP(3, [4, 4, 1])

# Example training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

# Training loop
for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward pass
    n.zero_grad()
    loss.backward()
    
    # Update
    for p in n.parameters():
        p.data += -0.1 * p.grad
    
    print(f'step {k}: loss = {loss.data}')
```

## Implementation Details

- The `Value` class uses a topological sort for the backward pass, ensuring correct gradient computation order.
- The neural network implementation uses the `tanh` activation function.
- Gradient descent is implemented manually in the training loop.

## Educational Purpose

This implementation is designed for educational purposes. It provides insights into:

- How autograd engines work
- The mechanics of backpropagation
- Basic structure of neural networks

For production use or larger projects, consider using established libraries like PyTorch or TensorFlow.

## Acknowledgements

This implementation is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) project.
