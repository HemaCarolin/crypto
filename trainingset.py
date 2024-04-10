import pickle
import torch
import tenseal as ts
import pandas as pd
import random
from time import time
import json

# Optional imports
import numpy as np
import matplotlib.pyplot as plt

torch.random.manual_seed(73)
random.seed(73)
np.random.seed(73)

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # Delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

# def calculate_mean_std(x_train):
#     # Calculate mean and standard deviation for each feature
#     training_mean = x_train.mean(axis=0)
#     training_std = x_train.std(axis=0)
#     return training_mean, training_std

def heart_disease_data():
    data = pd.read_csv("./framingham.csv")
    # Drop rows with missing values
    data = data.dropna()
    # Drop some features
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    # Balance data
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    # Extract labels
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop(columns=['TenYearCHD'])
    # Standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    print(x.shape)
    return split_train_test(x, y)

def preprocess_data(input_data):
    data = pd.read_csv("./framingham.csv")
    # Drop rows with missing values
    data = data.dropna()
    # Drop some features
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    # Balance data
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    # Extract labels
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop(columns=['TenYearCHD'])
    # Standardize data
    input_data = (input_data - data.mean()) / data.std()
    return input_data

def random_data(m=1024, n=2):
    # Data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = random_data()
x_train, y_train, x_test, y_test = heart_disease_data()
# Calculate mean and standard deviation for the training data

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print("TRAIN SAMPLE: ", x_train[10])
print(f"y_train has shape: {y_train.shape}")
print("TRAIN SAMPLE: ", y_train[10])
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")


class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out


class PlainLR(torch.nn.Module):

    def __init__(self, n_features):
        super(PlainLR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out


n_features = x_train.shape[1]
plain_model = PlainLR(n_features)

# Use gradient descent with a learning_rate=1
optim = torch.optim.SGD(plain_model.parameters(), lr=1)
# Use Binary Cross Entropy Loss
criterion = torch.nn.BCELoss()
# Define the number of epochs for both plain and encrypted training
EPOCHS = 5


def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model


plain_model = train(plain_model, optim, criterion, x_train, y_train)

# Function to save the plain PyTorch model
def save_plain_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

# After training the plain model, save it
save_plain_model(plain_model, "plain_model.pkl")
print('Saved plain model to "plain_model.pkl"')

def accuracy(model, x, y):
    out = model(x)
    # print("OUT:", out)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()


plain_accuracy = accuracy(plain_model, x_test, y_test)
print(f"Accuracy on plain test_set: {plain_accuracy}")

# parameters
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
# create TenSEALContext
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 21
ctx_eval.generate_galois_keys()


class EncryptedLR:

    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        # we accumulate gradients and count the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should run at least one forward iteration")
        # update weights
        # We use a small regularization term to keep the output
        # of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])

    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()

    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def save(self, filepath):
        """Saves the encrypted weights and bias to a custom binary file."""
        with open(filepath, 'wb') as file:
            # Don't decrypt, directly serialize the CKKS vectors
            pickle.dump([self.weight, self.bias], file)

    @staticmethod
    def load(filepath, context, torch_lr):
        """Loads the encrypted weights and bias from a custom binary file."""
        with open(filepath, 'rb') as file:
            # Load the list containing encrypted weights and bias (CKKS vectors)
            encrypted_params = pickle.load(file)
            weight, bias = encrypted_params

            # Convert loaded weights and bias back to CKKS vectors
            weight = ts.ckks_vector(context, weight)
            bias = ts.ckks_vector(context, bias)
            print("weight bias")
            print(weight)
            print(bias)
            # Create a new EncryptedLR instance with loaded parameters
            model = EncryptedLR(torch_lr)
            model.weight = weight
            model.bias = bias
            return model


eelr = EncryptedLR(plain_model)
eelr.save("encrypted_model.pkl")
print('Saved encrypted model parameters to "encrypted_model.pkl"')

# Now you can use the `EncryptedLR.load()` method to load the encrypted model from "encrypted_model.pkl"