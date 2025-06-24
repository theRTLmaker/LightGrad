# 🚀 LightGrad: Lightweight Autograd Engine for Neural Networks

A lightweight, intuitive autograd engine inspired by Karpathy’s micrograd, featuring extended operations, optimization algorithms, and visualization tools for deeper understanding and easier debugging of neural networks.

LightGrad is a minimalist, educational deep-learning library inspired by Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd). It extends core autograd capabilities with additional mathematical operations, modern optimization algorithms, and user-friendly visualization tools, designed for simplicity and ease of understanding.

Whether you’re learning deep learning concepts from scratch, experimenting with new ideas, or debugging neural network operations, LightGrad provides a transparent and powerful toolkit.

## 🌟 Key Features

* Core Autograd Engine:
  
Simple and intuitive automatic differentiation inspired by [micrograd](https://github.com/karpathy/micrograd).


*	Extended Mathematical Operations: Implementations of essential activation functions and loss metrics:
    * Activation Functions: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, and more.

* Loss Functions: MSE, Cross-Entropy.
Modern Optimization Algorithms:
    * SGD, SGD with Momentum, Adam, and RMSProp.
*	Visualization & Debugging Utilities:
	  *	Computation graph visualizer using Graphviz.
	  *	Gradient checking utilities to ensure correctness.

## 📦 Installation
Clone the repository and install the dependencies:
````
git clone https://github.com/theRTLmaker/LightGrad.git
cd LightGrad
pip install -r requirements.txt
````

## 🎓 Quickstart Example
Here’s a quick example of building and training a simple neural network with LightGrad:
````
from lightgrad.engine import Value
from lightgrad.nn import MLP
from lightgrad.optim import Adam

# Simple dataset (XOR example)
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

# Create an MLP (Multi-Layer Perceptron)
model = MLP(2, [4, 4, 1])  

# Adam optimizer
optim = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # forward pass
    y_pred = [model(xi)[0] for xi in X]

    # compute loss (MSE)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(y, y_pred))

    # backward pass and optimization
    optim.zero_grad()
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch} loss: {loss.data}')
````

## 🖥️ Visualization & Debugging
Generate computation graph visualizations:
````
from lightgrad.visualize import draw_dot

# Example forward pass
x = Value(2.0)
y = Value(3.0)
z = x * y + y**2

# Draw and visualize
draw_dot(z).render('graph', format='png', view=True)
````
Perform gradient checking:
````
from lightgrad.debug import gradient_check

# define a function
def f(x):
    return x * x + 2 * x + 1

# check gradients at x = 2.0
gradient_check(f, 2.0)
````

## 🛠️ Project Structure
````
LightGrad/
├── lightgrad/
│   ├── engine.py            # Core autograd engine
│   ├── nn.py                # Neural network layers
│   ├── optim.py             # Optimizers
│   ├── losses.py            # Loss functions
│   ├── activations.py       # Activation functions
│   ├── visualize.py         # Visualization utilities
│   └── debug.py             # Gradient checking utilities
├── tests/                   # Unit tests
├── examples/                # Practical examples
├── requirements.txt         # Python dependencies
└── README.md
````

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments
*	Inspired by Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd), which provided the foundational ideas for this project.

##	💬 Contributing
Contributions are welcome! Open an issue or submit a pull request to discuss enhancements, bug fixes, or feature requests.
