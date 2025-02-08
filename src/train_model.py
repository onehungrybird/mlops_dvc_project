# src/train_model.py
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    X, y = joblib.load("data/mnist.pkl")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_regression.pkl")

if __name__ == "__main__":
    train_model()