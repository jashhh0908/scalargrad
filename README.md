# ScalarGrad

ScalarGrad is a pure Python, scalar-valued automatic differentiation engine and neural network library built from first principles as a learning project to master the fundamental mechanics of Deep Learning and systems-level graph traversal. Designed without reliance on high-level tensor frameworks (like PyTorch or NumPy), ScalarGrad implements a dynamic computational graph to perform reverse-mode automatic differentiation (backpropagation) and gradient-based optimization from the ground up.

## Core Systems & Mathematical Architecture

- **Dynamic Directed Acyclic Graph (DAG):** Employs Python operator overloading (`+`, `-`, `*`, `**`, `relu`) to dynamically generate an Intermediate Representation (IR) of the mathematical expression as it is executed.
- **Reverse-Mode AD & Multivariate Chain Rule:** Encodes analytical local derivatives as closures within each node. Gradients are accumulated (`+=`) to satisfy the multivariate chain rule across diverging branches.
- **Topological Traversal:** Implements a post-order topological sort via Depth-First Search (DFS) to traverse the DAG in reverse, guaranteeing that a node's global gradient is fully computed before propagating to its children.
- **Neural Network Abstractions:** Provides `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) primitives. Supports non-linear activations (`ReLU`) and configurable linear output heads to prevent gradient death on continuous regression tasks.

## Quickstart & Usage

```python
from engine import Value
from nn import MLP

# 1. Initialize a Multi-Layer Perceptron (3 inputs, two hidden layers of 4, 1 output)
model = MLP(3, [4, 4, 1])

# 2. Define inputs and target
x = [2.0, 3.0, -1.0]
target = 1.0

# 3. Forward Pass & Loss Calculation (Mean Squared Error)
prediction = model(x)
loss = (prediction - target) ** 2

# 4. Zero gradients and execute Backward Pass
for p in model.parameters():
    p.grad = 0.0
loss.backward()

# 5. Gradient Descent Update (Learning Rate = 0.01)
for p in model.parameters():
    p.data -= 0.01 * p.grad

print(f"Prediction: {prediction.data:.4f} | Loss: {loss.data:.4f}")
```



## Project Structure

```text
scalargrad/
├── engine.py        # Core Autodiff engine, Value scalar object, and DAG operations
├── nn.py            # Neural network primitives (Neuron, Layer, MLP) and parameter management
└── README.md
```
