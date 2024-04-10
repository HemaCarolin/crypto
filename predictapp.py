import pickle
import tenseal as ts
import numpy as np
import torch

# Suppose you have encrypted logistic regression model parameters
# Load the encrypted model parameters
def load_encrypted_model(filepath, context):
    with open(filepath, 'rb') as file:
        # Load the encrypted weights and bias (CKKS vectors)
        encrypted_params = pickle.load(file)
        weight, bias = encrypted_params

        # Convert loaded weights and bias back to CKKS vectors
        weight = ts.ckks_vector(context, weight)
        bias = ts.ckks_vector(context, bias)

        return weight, bias

# Load the context (previously created)
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
context.global_scale = 2 ** 21
context.generate_galois_keys()

# Load the encrypted logistic regression model parameters
encrypted_weight, encrypted_bias = load_encrypted_model("encrypted_model.pkl", context)

# Define the prediction function using the encrypted model
def predict_encrypted_model(encrypted_weight, encrypted_bias, input_data):
    # Convert the input data to a 1D array (vector)
    input_vector = np.array(input_data, dtype=np.float32)

    # Encrypt the input features
    encrypted_input = ts.ckks_vector(context, input_vector.tolist())

    # Perform the encrypted prediction
    encrypted_prediction = encrypted_input.dot(encrypted_weight) + encrypted_bias
    encrypted_prediction = encrypted_prediction.polyval([0.5, 0.197, 0, -0.004])  # Sigmoid activation function

    # Decrypt the prediction
    decrypted_prediction = encrypted_prediction.decrypt()

    # Since the decryption may return multiple values, take the first value
    decrypted_prediction = decrypted_prediction[0]

    # Convert the decrypted prediction to a binary value (0 or 1)
    binary_prediction = 1 if decrypted_prediction > 0.5 else 0
    
    return binary_prediction

# Example input data (features for heart disease prediction)
input_data = [0., 49., 0., 0., 0., 224., 130., 75., 73.]
input_data2 = [1., 50., 0., 0., 1., 240., 140., 80., 100.]


# Make a privacy-preserving prediction using the encrypted logistic regression model
prediction = predict_encrypted_model(encrypted_weight, encrypted_bias, input_data2)
print("Predicted label (0: No heart disease, 1: Heart disease):", prediction)
