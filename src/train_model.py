"""
Project 2: Data Classification Using AI
DecodeLabs - Artificial Intelligence Engineer Track

This script trains a K-Nearest Neighbors classifier on the Iris dataset
and saves the trained model and scaler to disk for later use in the
Streamlit deployment app (app.py).

Run with:
    python src/train_model.py
"""

import os
import pickle

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    # 1. Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # 4. Find optimal K
    error_rates = []
    k_range = range(1, 31)
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        pred_temp = knn_temp.predict(X_test)
        error_rates.append(np.mean(pred_temp != y_test))
    optimal_k = list(k_range)[int(np.argmin(error_rates))]

    # 5. Train final model
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, y_train)

    # 6. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"Optimal K: {optimal_k}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print()
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # 7. Save model and scaler
    os.makedirs("models", exist_ok=True)
    with open("models/knn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nSaved trained model to models/knn_model.pkl")
    print("Saved scaler to models/scaler.pkl")


if __name__ == "__main__":
    main()
