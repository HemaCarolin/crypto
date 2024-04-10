import base64
import gzip
import io
import pickle
from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
import numpy as np
import tenseal as ts
import pandas as pd
import torch
from sklearn import preprocessing
import trainingset as training  # Assuming trainingset.py contains the PlainLR and EncryptedLR classes

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/submit_form": {"origins": "http://localhost:3000"},
                     r"/predict": {"origins": "http://localhost:3000/user-datainput"}})

# Load the plain PyTorch model
def load_plain_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the encrypted model from the saved file
def load_encrypted_model(filepath, context, torch_lr):
    with open(filepath, 'rb') as file:
        # Load the encrypted weights and bias (CKKS vectors)
        encrypted_params = pickle.load(file)
        weight, bias = encrypted_params

        # Convert loaded weights and bias back to CKKS vectors
        weight = ts.ckks_vector(context, weight)
        bias = ts.ckks_vector(context, bias)

        # Create a new EncryptedLR instance with loaded parameters
        model = training.EncryptedLR(torch_lr)
        model.weight = weight
        model.bias = bias
        return model

poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 21

# Generate Galois keys
ctx_eval.generate_galois_keys()

# Define the path to the plain PyTorch model file
PLAIN_MODEL_FILEPATH = "plain_model.pkl"
ENCRYPTED_MODEL_FILEPATH = "encrypted_model.pkl"

# Load the plain PyTorch model
# model = load_plain_model(PLAIN_MODEL_FILEPATH)

# Load the encrypted model
# encrypted_model = load_encrypted_model(ENCRYPTED_MODEL_FILEPATH, ctx_eval, model)


def predict_single(model, x):
    # Set the model to evaluation mode
    model.eval()
    
    # Convert input data to a PyTorch tensor
    x_tensor = x.clone().detach().unsqueeze(0).float()  # Add an extra dimension for batch size
    
    # Make prediction
    with torch.no_grad():
        prediction = model(x_tensor)
        # Convert prediction to binary (0 or 1) based on threshold 0.5
        print("ACTUAL PRED: ", prediction)
        binary_prediction = 1 if prediction.item() >= 0.5 else 0
    
    return binary_prediction


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    plain_model = training.PlainLR(training.n_features)
    # Use gradient descent with a learning_rate=1
    optim = torch.optim.SGD(plain_model.parameters(), lr=0.0001)
    # Use Binary Cross Entropy Loss
    criterion = torch.nn.BCELoss()
    model = training.train(plain_model, optim, criterion, training.x_train, training.y_train)

    # model = load_plain_model(PLAIN_MODEL_FILEPATH)

    # Convert string values to floats in the data dictionary
    input_data = {key: float(value) if value is not None else 0.0 for key, value in data.items()}
    print("input_data")
    print(input_data)
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    print("input_df")
    print(input_df)
    # Drop some features
    input_df = input_df.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    print("input_df")
    print(input_df)
    # Convert the DataFrame to a tensor
    input_df = training.preprocess_data(input_df)
    input_tensor = torch.tensor(input_df.values).float()
    print("input_DF: ")
    print(input_tensor)
    print(input_tensor.shape)
    # Perform prediction using the loaded plain model
    # with torch.no_grad():
    #     out = model(input_tensor)
    # print("OUT: ", out)


    # # Round the output to get binary predictions
    # predictions = model(input_tensor)
    # print("Pred Tensor: ")
    # print(predictions)

    #given_list = [0., 49., 0., 0., 0., 224., 130., 75., 73.]

    # Convert the list into a PyTorch tensor
    # tensor_from_list = torch.tensor(given_list)
    # print("STD DATA: ", standardized_data)
    out = predict_single(model, input_tensor)
    # print("OUT: ", out)

    # Return the predictions as JSON
    return jsonify({"predictions": out})

if __name__ == '__main__':
    app.run(debug=True)
