import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Optional

class MLP:
    """
    A Multilayer Perceptron for Regression and Classification Tasks
    """
    def __init__(
        self, 
        num_inputs : int, 
        num_hidden : int, 
        num_outputs : int, 
        task : str = 'regression', 
        seed : Optional[int] = None
    ):
        """
        Initializes the Perceptron for a given task

        Parameters:
            num_inputs (int): The number of inputs
            num_hidden (int): The number of hidden units
            num_outputs (int): The number of outputs
            task (String): The Perceptron's intended task, either 'regression' or 'classification'

        """
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be either 'regression' or 'classification'")
        
        if seed is not None:
            np.random.seed(seed)
        
        self.task = task
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Xavier initialization for weights
        hidden_bounds = np.sqrt(6 / (self.num_inputs + self.num_hidden))
        output_bounds = np.sqrt(6 / (self.num_hidden + self.num_outputs))
        self.hidden_weights = np.random.uniform(-hidden_bounds,hidden_bounds,(self.num_hidden, self.num_inputs))
        self.output_weights = np.random.uniform(-output_bounds,output_bounds,(self.num_outputs, self.num_hidden))

        self.hidden_biases = np.zeros(num_hidden)
        self.output_biases = np.zeros(num_outputs)

        self.reset_gradients()

    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the MLP.

        Parameters:
            inputs (np.ndarray): Input array of shape (num_inputs,).

        Returns:
            np.ndarray: Output activations after processing through hidden and output layers.
        """
        self.hidden_activations = np.tanh(np.dot(self.hidden_weights, inputs) + self.hidden_biases)
        self.output_activations = np.dot(self.output_weights, self.hidden_activations) + self.output_biases

        # Apply softmax activation if the task is classification
        if self.task == 'classification':
            self.output_activations = self.softmax(self.output_activations)
        return self.output_activations

    def backpropagate(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Performs the backward pass of the MLP by computing errors and accumulating weight deltas.

        Parameters:
            inputs (np.ndarray): Input array of shape (num_inputs,).
            targets (np.ndarray): Target array of shape (num_outputs,).

        Raises:
            ValueError: If the task is classification but the error term is incorrectly calculated.
        """
        # Compute errors
        if self.task == 'classification':
            # Correct error term for classification
            output_errors = self.output_activations - targets
        else:
            # Correct error term for regression (MSE)
            output_errors = (self.output_activations - targets) * 2  # Derivative of MSE

        hidden_errors = np.dot(self.output_weights.T, output_errors)
        hidden_deltas = hidden_errors * self.d_tanh(self.hidden_activations)  # tanh derivative

        # Accumulate weight deltas
        self.output_weight_grad += np.outer(output_errors, self.hidden_activations)
        self.output_bias_grad += output_errors
        self.hidden_weight_grad += np.outer(hidden_deltas, inputs)
        self.hidden_bias_grad += hidden_deltas

    def d_tanh(self, x) -> float:
        return (1 - x ** 2)

    def update_weights(self, learning_rate: float) -> None:
        """
        Updates the weights and biases using the accumulated gradients.

        Parameters:
            learning_rate (float): The learning rate for weight updates.
            gradient_threshold (float): The threshold for gradient clipping.
        """
        self.clip_gradients()
        self.gradient_descent(learning_rate)
        self.reset_gradients()

    def gradient_descent(self, learning_rate: float) -> None:
        """
        Update the weights based on the accumulated gradients

        Parameters:
            learning_rate (float): The learning rate for weight updates
        """
        self.output_weights -= learning_rate * self.output_weight_grad
        self.output_biases -= learning_rate * self.output_bias_grad
        self.hidden_weights -= learning_rate * self.hidden_weight_grad
        self.hidden_biases -= learning_rate * self.hidden_bias_grad


    def clip_gradients(self, gradient_threshold: float = 1.0) -> None:
        """
        Clips gradients to avoid explosions

        Parameters:
            gradient_threshold (float): the maximum allowed magnitude of a gradient
        """
        self.output_weight_grad = np.clip(self.output_weight_grad, -gradient_threshold, gradient_threshold)
        self.output_bias_grad = np.clip(self.output_bias_grad, -gradient_threshold, gradient_threshold)
        self.hidden_weight_grad = np.clip(self.hidden_weight_grad, -gradient_threshold, gradient_threshold)
        self.hidden_bias_grad = np.clip(self.hidden_bias_grad, -gradient_threshold, gradient_threshold)

    def reset_gradients(self) -> None:
        """
        Sets all gradients to zero
        """
        self.hidden_weight_grad = np.zeros_like(self.hidden_weights)
        self.hidden_bias_grad = np.zeros_like(self.hidden_biases)
        self.output_weight_grad = np.zeros_like(self.output_weights)
        self.output_bias_grad = np.zeros_like(self.output_biases)

    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int, 
        learning_rate: float, 
        batch_size: int, 
        decay_rate: float = 1, 
        learning_update_interval: int = 50
    ) -> List[float]:
        """
        Trains the MLP on the provided dataset.

        Parameters:
            X_train (np.ndarray): Input data. This method will try to cast input to an np.ndarray
            labels (np.ndarray): Target data. This method will try to cast input to an np.ndarray
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
            batch_size (int): Number of samples per weight update.
            decay_rate (float, optional): Factor by which to decay the learning rate every `weight_update_interval` epochs. Default is 1 (no decay).
            weight_update_interval (int, optional): Interval (in epochs) at which to apply learning rate decay and print debug information. Default is 50.

        Returns:
            List[float]: A list containing the mean error for each epoch.
        """
        X_train = np.array(X_train.copy())
        y_train = np.array(y_train.copy())
        errors: List[float] = []
        
        for epoch in range(epochs):
            # Shuffle dataset
            shuffled_indices = np.random.permutation(len(X_train))
            X_train = X_train[shuffled_indices]
            y_train = y_train[shuffled_indices]

            targets = y_train

            epoch_error = 0.0
            for i, (inputs, target) in enumerate(zip(X_train, targets)):
                self.forward(inputs)
                self.backpropagate(inputs, target)

                if self.task == 'regression':
                    # Compute Mean Squared Error (MSE)
                    epoch_error += np.mean((target - self.output_activations) ** 2)
                elif self.task == 'classification':
                    # Compute Cross-Entropy Loss
                    epoch_error += -np.sum(target * np.log(self.output_activations))

                # Update weights periodically based on batch_size
                if (i + 1) % batch_size == 0 or i == len(X_train) - 1:
                    self.update_weights(learning_rate)
            
            mean_error = epoch_error / len(X_train)
            errors.append(mean_error)

            # Apply learning rate decay at specified intervals
            if epoch % learning_update_interval == 0 and epoch > 0:
                learning_rate *= decay_rate

            # Print training updates at each weight update interval
            if (epoch + 1) % learning_update_interval == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Error: {mean_error}")
        
        return errors
            
    def softmax(self, X: np.ndarray) -> np.ndarray:
        """
        Helper method for calculating the softmax 
        """
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum()
