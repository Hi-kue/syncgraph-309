import os
import sys
import pickle
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.orouter_client import OpenRouterClient
from model.smote_type import SmoteType

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenRouterClient()


def format_classification_report(y_test, y_pred) -> None:
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(
        report_df.style.format("{:.2f}")
        .highlight_max(subset=['precision', 'recall', 'f1-score'], color='#0099ff'),
        use_container_width=True
    )


def display_model_performance(model, x_test, y_test, model_name) -> None:
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
    plt.title(f'Confusion Matrix - {model_name}')
    st.pyplot(plt)

    format_classification_report(
        y_test=y_test,
        y_pred=y_pred
    )


def safe_load_models(model_path) -> Any | None:
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
            return model

    except Exception as e:
        st.error(f"Could not load model: {e}")


def prediction_analysis(input_json: dict, model_name: str, _type: str) -> None:
    st.markdown("---")
    st.subheader(f"Prediction Analysis for {model_name}")

    print(f"input_json: {input_json}")

    smote_type = SmoteType.match_str(_type)

    response = client.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"""
                The predictive model used is `{model_name}`.
                
                Here is the JSON data you will make an analysis on: 
                {input_json}
                
                The oversampling technique that was used for this analysis is `{smote_type}`.
                """
            }
        ],
        temperature=1,
        max_tokens=0
    )

    if response:
        st.markdown(body=f"""
             <div style="word-wrap: break-word; overflow-wrap: break-word; white-space: normal;">
                {response}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("There was an error processing the request you provided.")
