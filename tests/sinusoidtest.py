import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from MLP import MLP

# for determining the MSE in the test
def mean_squared_error(y_true, y_pred):
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

def test_sinusoid_problem() -> None:
    # 500 4-element vectors containing random values between -1 and 1 
    X = np.random.uniform(-1, 1, (500, 4))

    # calculate the 500 targets
    y = []
    for nums in X:
        x1, x2, x3, x4 = nums
        y.append(np.sin(x1-x2+x3-x4))

    # split the 500 samples into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create an MLP and train it on the training set 
    mlp = MLP(num_inputs=4, num_hidden=10, num_outputs=1, task = 'regression')
    errors = mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=10, decay_rate=0.9, learning_update_interval=50)

    # use the MLP to predict the outputs for the test set
    y_pred = []
    for inputs in X_test:
        output = mlp.forward(inputs)
        y_pred.append(output)

    # print the test error
    print(mean_squared_error(y_test, y_pred))

    # plot the MSE as a function of the epoch
    plt.plot(errors)
    plt.title('Error During Training for Sinusoid Problem')
    plt.xlabel('Epoch')
    plt.ylabel('Error (MSE)')
    plt.show()

    # plot the predictions and targets
    plt.scatter(y_test, y_pred)
    plt.title('Sinusoid Approximation Results')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.show()

if __name__ == "__main__":
    test_sinusoid_problem()