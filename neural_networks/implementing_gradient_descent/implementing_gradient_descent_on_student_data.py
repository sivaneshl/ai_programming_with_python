import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field]-mean)/std

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
train_data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = train_data.drop('admit', axis=1), train_data['admit']
test_features, test_targets = test_data.drop('admit', axis=1), test_data['admit']

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

n_recocrds, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1/n_features**0.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)

    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Note: We haven't included the h variable from the previous
        #       lesson. You can add it if you want, or you can calculate
        #       the h together with the output

        # Calculate the output
        h = np.dot(x, weights)
        y_hat = sigmoid(h)

        # Calculate the error
        error = y-y_hat

        # Calculate the error term
        # error_term = error * y_hat * (1-y_hat)
        error_term = error * sigmoid_prime(h)

        # Calculate the change in weights for this sample and add it to the total weight change
        del_w += error_term * x

    # Update weights using the learning rate and the average change in weights
    weights += learnrate * del_w / n_recocrds

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
test_out = sigmoid(np.dot(test_features, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == test_targets)
print("Prediction accuracy: {:.3f}".format(accuracy))


