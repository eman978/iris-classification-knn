"""
Project 2: Data Classification Using AI
DecodeLabs - Artificial Intelligence Engineer Track

Streamlit deployment app for the trained KNN Iris classifier.

Run locally with:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud by connecting this
GitHub repository at https://share.streamlit.io
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

MODEL_PATH = os.path.join("models", "knn_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

st.set_page_config(
    page_title="Iris Classifier | DecodeLabs Project 2",
    page_icon=None,
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def main():
    st.title("Iris Species Classifier")
    st.caption("Project 2: Data Classification Using AI — DecodeLabs")

    st.write(
        "This app uses a K-Nearest Neighbors model trained on the Iris "
        "dataset to classify a flower into one of three species based on "
        "four measurements."
    )

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.error(
            "Model files not found. Please run `python src/train_model.py` "
            "first to generate models/knn_model.pkl and models/scaler.pkl."
        )
        return

    model, scaler = load_artifacts()
    iris = load_iris()

    st.subheader("Enter Flower Measurements")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

    if st.button("Classify", type="primary"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        species_name = iris.target_names[prediction]
        st.success(f"Predicted Species: **{species_name}**")

        prob_df = pd.DataFrame(
            {"Species": iris.target_names, "Probability": probabilities}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Species"))

    with st.expander("About this model"):
        st.write(
            "- Algorithm: K-Nearest Neighbors (KNN)\n"
            "- Dataset: Iris (150 samples, 3 classes, 4 features)\n"
            "- Preprocessing: StandardScaler feature scaling\n"
            "- Split: 80% training / 20% testing\n"
            "- Evaluation: Confusion matrix, precision, recall, F1 score "
            "(see notebooks/Project2_Data_Classification_Using_AI.ipynb "
            "for the full analysis)"
        )


if __name__ == "__main__":
    main()
