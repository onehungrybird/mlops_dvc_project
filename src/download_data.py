# src/download_data.py
import os
from sklearn.datasets import fetch_openml
import joblib
print("==========")

def download_mnist():
    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]

    # Save dataset locally
    os.makedirs("data", exist_ok=True)
    joblib.dump((X, y), "data/mnist.pkl")

if __name__ == "__main__":
    download_mnist()