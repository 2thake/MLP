import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from MLP import MLP

def test_xor_problem() -> None:
    xor_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    xor_targets = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    # Create the MLP and set it up for regression
    mlp = MLP(num_inputs=2, num_hidden=3, num_outputs=2, task='classification')

    # train the MLP and store the errors
    errors = mlp.train(xor_inputs, xor_targets, epochs=1000, learning_rate=0.1, batch_size=1)

    # print the results
    print("\nTesting XOR Problem:")
    for inputs, target in zip(xor_inputs, xor_targets):
        outputs = mlp.forward(inputs)
        print(f"Input: {inputs}, Predicted: {outputs}, Target: {target}")

    # plot the error magnitude by epoch
    plt.plot(errors)
    plt.title('Error During Training for XOR Problem')
    plt.xlabel('Epoch')
    plt.ylabel('Error (Cross-Entropy Loss)')
    plt.show()


if __name__ == "__main__":
    test_xor_problem()