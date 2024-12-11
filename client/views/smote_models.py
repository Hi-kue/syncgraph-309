import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.smote_type import SmoteType
from core.model_funcs import (
    safe_load_models,
    display_model_performance,
    format_classification_report,
    prediction_analysis
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# region Pre-work for Models
CURRENT_DIR = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
MODELS_DIR = os.path.join(ROOT, "server", "models")
DATA_DIR = os.path.join(ROOT, "server", "data")


# NOTE: Safe loading models from ../../server/models folder
lr_model = safe_load_models(os.path.join(MODELS_DIR, "lr_model.pkl"))
dt_model = safe_load_models(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf_model = safe_load_models(os.path.join(MODELS_DIR, "rf_model.pkl"))

toodu_df = pd.read_csv(os.path.join(DATA_DIR, "Theft_Over_Open_Data_Cleaned.csv"))


def create_selectbox_options(label_encoder) -> list[tuple[str, int]]:
    return [(f"({encoded}): {title}", encoded) for encoded, title in enumerate(label_encoder.classes_)]


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

premises_options = create_selectbox_options(premises_le)
location_options = create_selectbox_options(location_le)

x = toodu_df[categorical_features]
y = toodu_df["OFFENCE_ENCODED"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

location_type = list(toodu_df["LOCATION_TYPE"].unique())
premises_type = list(toodu_df["PREMISES_TYPE"].unique())
# endregion


st.set_page_config(
    page_title="Theft Prediction Analysis",
    page_icon="📈",
    layout="centered"
)

# region Sidebar Configuration
models = {
    "Logistic Regression (SMOTE)": lr_model,
    "Decision Tree (SMOTE)": dt_model,
    "Random Forest (SMOTE)": rf_model,
}
model_names = list(models.keys())

st.sidebar.title("Discovery Sidebar 🌐")
st.sidebar.selectbox("Select a Model to View and Use for Predictions:", model_names, key="model")
selected_model = models[st.session_state.model]
# endregion

st.title("Theft Prediction Analysis 📈")
st.write("""
## Predictive Modeling for Theft Risk Assessment

This dashboard allows you to explore the various predictive and
classifications models we used to predict the likelihood of whether
a `LOCATION_TYPE` and `PREMISES_TYPE` was an *Auto Theft* or a *Non-Auto Theft*.
""")

st.write("""
On the left side where you see the sidebar, you can select the model you want to view.
""")
st.markdown("---")

display_model_performance(selected_model, x_test, y_test, st.session_state.model)

st.markdown("---")

st.header("Model Predictor In Action")
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

user_input_smote = st.session_state.get("user_inputs_smote", [])
entry_format = {
    "LOCATION_TYPE": st.selectbox("LOCATION_TYPE", options=location_options, format_func=lambda n: n[0]),
    "PREMISES_TYPE": st.selectbox("PREMISES_TYPE", options=premises_options, format_func=lambda n: n[0]),
}

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Entries", use_container_width=True):
            user_input_smote.clear()
            st.session_state.user_inputs_smote = user_input_smote
            st.toast("All Entries have been Cleared!", icon="✅")

    with col2:
        if st.button("Add Entry", use_container_width=True):
            user_input_smote.append({k: v[1] for k, v in entry_format.items()})
            st.session_state.user_inputs_smote = user_input_smote
            st.toast("Entry has been Successfully Added!", icon="✅")

with st.expander("Current JSON Request", expanded=True):
    st.write("""
    Your current request looks like the following:
    """)
    st.json(user_input_smote)

st.write("""
> **NOTE**:
> The above JSON object is dynamically generated so long as you interact and add more entries.

Once you are satisfied with the entries you have added, click on the `Predict` button below to make the
predictions based on the entries you provided above.
""")
if st.button("Predict Probabilities 🔮", use_container_width=True):
    st.balloons()
    st.session_state.user_inputs_smote = user_input_smote
    predictions = requests.post(
        "http://localhost:5000/api/v1/predict",
        params={"model_name": str(selected_model.__class__.__name__)},
        json=st.session_state.user_inputs_smote,
    ).json()

    st.write(predictions)

    if predictions["status"] == 200:
        st.write(f"""
        Predictions made on `{predictions['timestamp']}` 📅,
        using the `{predictions['model']}` model.
        
        The model is `{predictions['prediction']['confidence']}` confident of its predictions.
        """)
    else:
        st.write(f"Error: {predictions['data']['error']}")

    prediction_analysis(predictions, st.session_state.model, SmoteType.SMOTE.value[0])
