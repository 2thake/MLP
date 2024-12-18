# MLP (Multilayer Perceptron)

This project implements a simple **Multilayer Perceptron (MLP)** in Python. The `MLP` package contains the source for the perceptron, and the tests package contains a number of files which test the perceptron on elementary machine learning tasks.

## Features
- **Activation Functions**
  - **Hidden Layer**: Employs the hyperbolic tangent (`tanh`) activation function for non-linear transformations.
  - **Output Layer**: Uses either linear or softmax activation functions depending on the chosen task.

- **Training Details**
  - **Forward and Backward Propagation**: Implements both forward pass for predictions and backward pass for gradient computation.
  - **Gradient Descent**: Updates weights and biases using the gradient descent optimization algorithm with a configurable learning rate.
  - **Gradient Accumulation**: Accumulates gradients over mini-batches to optimize weight updates effectively.
  - **Gradient Clipping**: Prevents gradient explosion by clipping gradients to a specified threshold.
  - **Learning Rate Decay**: Optionally decays the learning rate at specified intervals to fine-tune the training process.
  - **Xavier Initialization**: Utilizes Xavier (Glorot) initialization for both hidden and output layer weights to promote efficient training and convergence.

- **Utility Functions**
  - **Softmax Implementation**: Includes a helper method for calculating the softmax function, essential for classification tasks.
  - **Error Metrics**: Computes appropriate error metrics such as Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/2thake/MLP.git
   cd MLP

2. Create and activate a virtual environment
   ```python
   python -m venv venv
    source venv/bin/activate   # On Linux/macOS
    venv\Scripts\activate      # On Windows

3. Install dependencies (Most of these are used for the tests. The MLP implementation only needs numpy.)
   ```python
    pip install -r requirements.txt

### Running Tests
Test the functionality of the `MLP` package with provided test scripts. Run the test scripts as modules from the project root directory:

1. **XOR Problem:**
   ```bash
   python -m tests.xortest

2. **Sinusoidal Approximation:**
   ```bash
    python -m tests.sinusoidtest

3. **Letter Recognition:**
   ```bash
   python -m tests.lettertest

## Using the MLP Package
To use the `MLP` package in your own scripts, import it as follows:
   ```python
    from MLP import MLP

    # Create and train an MLP instance
    model = MLP(input_size=2, hidden_size=4, output_size=2)
    model.train(inputs, targets)
