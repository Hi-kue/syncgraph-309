import os
import sys
import pickle
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rich_logging import logger as log
from core.orouter_client import OpenRouterClient

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# region Pre-work for Models
CURRENT_DIR = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
MODELS_DIR = os.path.join(ROOT, "syncgraph-309", "server", "models")
DATA_DIR = os.path.join(ROOT, "syncgraph-309", "server", "data")


def safe_load_models(model_path):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


# NOTE: Safe loading models from ../../server/models folder
lr_model = safe_load_models(os.path.join(MODELS_DIR, "lr_model.pkl"))
dt_model = safe_load_models(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf_model = safe_load_models(os.path.join(MODELS_DIR, "rf_model.pkl"))


toodu_df = pd.read_csv(os.path.join(DATA_DIR, "Theft_Over_Open_Data_Cleaned.csv"))

theft_over_categories = {
    "Theft - Misapprop Funds Over",
    "Theft Over - Bicycle",
    "Theft Over - Distraction",
    "Theft Over",
    "Theft Over - Shoplifting",
    "Theft Of Utilities Over",
    "Theft From Mail / Bag / Key"
}
toodu_df["OFFENCE"] = toodu_df["OFFENCE"].replace(theft_over_categories, "Theft Over")

le = LabelEncoder()
premises_le = LabelEncoder()
location_le = LabelEncoder()

toodu_df["PREMISES_TYPE"] = premises_le.fit_transform(toodu_df["PREMISES_TYPE"])
toodu_df["LOCATION_TYPE"] = location_le.fit_transform(toodu_df["LOCATION_TYPE"])
toodu_df["OFFENCE_ENCODED"] = le.fit_transform(toodu_df["OFFENCE"])

categorical_features = [
    "PREMISES_TYPE",
    "LOCATION_TYPE"
]

for col in categorical_features:
    toodu_df[col] = LabelEncoder().fit_transform(toodu_df[col])

premises_options = [
    (f"({encoded}): {title}", encoded) for encoded, title in enumerate(premises_le.classes_)
]
location_options = [
    (f"({encoded}): {title}", encoded) for encoded, title in enumerate(location_le.classes_)
]

x = toodu_df[categorical_features]
y = toodu_df["OFFENCE_ENCODED"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

location_type = list(toodu_df["LOCATION_TYPE"].unique())
premises_type = list(toodu_df["PREMISES_TYPE"].unique())
# endregion


def format_classification_report(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(
        report_df.style.format("{:.2f}")
        .highlight_max(subset=['precision', 'recall', 'f1-score'], color='#0099ff'),
        use_container_width=True
    )


def display_model_performance(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cv_scores = cross_val_score(model, x_test, y_test, cv=5)

    st.subheader(f"{model_name} Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}", f"{accuracy:.2%}")
    col1.metric("Cross-Validation Mean Score", f"{cv_scores.mean():.2%}")
    col2.metric("Precision", f"{precision:.2%}", f"{precision:.2%}")
    col2.metric("Cross-Validation Score Std", f"{cv_scores.std():.2%}")
    col3.metric("Recall", f"{recall:.2%}", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}", f"{f1:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    st.pyplot(plt)

    format_classification_report(
        y_test=y_test,
        y_pred=y_pred
    )


def main():
    st.set_page_config(
        page_title="Theft Prediction Analysis",
        page_icon="📈",
        layout="centered"
    )

    st.title("Theft Prediction Analysis Dashboard")
    st.write("""
    ## Predictive Modeling for Theft Risk Assessment
    
    This dashboard allows you to explore the various predictive and
    classifications models we used to predict the likelihood of whether
    a `LOCATION_TYPE` and `PREMISES_TYPE` was an *Auto Theft* or a *Non-Auto Theft*.
    """)
    st.markdown("---")

    models = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }
    model_names = list(models.keys())

    st.sidebar.title("Discovery Sidebar 🌐")
    st.sidebar.markdown("---")
    st.sidebar.selectbox("Select a Model to View:", model_names, key="model")
    selected_model = models[st.session_state.model]

    display_model_performance(selected_model, x_test, y_test, st.session_state.model)

    st.markdown("---")

    st.header("Model Preditor In Action")
    st.write("""
    **What is this section about?**
    
    For this section, we will be using the `predict` endpoint to make predictions using
    a selected model. The predictions will be determined based on the input you provide
    in the form of a JSON object. The JSON object will be dynamically generated as you 
    interact with the input fields on the left side of the page (the sidebar). 
    
    To refer to the endpoints, they are the following:
    1. `http://localhost:5000/api/v1/predict` - This is the endpoint for model predictions.
    2. `http://localhost:5000/api/v1/summarize` - This is the endpoint for model summaries using AI.
    3. `http://localhost:5000/api/v1/` - This is a general endpoint for the API.
    """)

    user_input = st.session_state.get("user_inputs", [])
    entry_format = {
        "LOCATION_TYPE": st.selectbox("LOCATION_TYPE", options=location_options, format_func=lambda n: n[0]),
        "PREMISES_TYPE": st.selectbox("PREMISES_TYPE", options=premises_options, format_func=lambda n: n[0]),
    }

    if st.button("Add Entry"):
        user_input.append({k: v[1] for k, v in entry_format.items()})
        st.session_state.user_inputs = user_input

    st.write("""
    Your current request looks like the following:
    """)
    st.json(user_input)

    st.write("""
    > [!NOTE
    > The above JSON object is dynamically generated so long as you interact and add more entries.
    
    Once you are satisfied with the entries you have added, click on the `Predict` button below to make the
    predictions based on the entries you provided above.
    """)
    if st.button("Predict Probabilities 🔮"):
        st.session_state.user_inputs = user_input
        predictions = requests.post(
            "http://localhost:5000/api/v1/predict",
            json=user_input
        ).json()

        st.write(predictions)

        if predictions["status"] == 200:
            st.write(f"""
            Predictions made on `{predictions['timestamp']}` 📅,
            using the `{predictions['model']}` model.
            
            The model is `{predictions['prediction']['confidence']}` confident of its predictions.
            """)
        else:
            st.write(f"Error: {predictions['error']}")


if __name__ == "__main__":
    main()
