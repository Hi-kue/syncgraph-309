import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_funcs import (
    safe_load_models,
    display_model_performance,
    prediction_analysis
)

import pandas as pd
import pydeck as pdk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# region Pre-work for Models
CURRENT_DIR = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
MODELS_DIR = os.path.join(ROOT, "server", "models")
DATA_DIR = os.path.join(ROOT, "server", "data")


# NOTE: Safe loading models from ../../server/models folder (ONLY SMOTENC)
lr_model_smotenc = safe_load_models(os.path.join(MODELS_DIR, "lr_model_smotenc.pkl"))
dt_model_smotenc = safe_load_models(os.path.join(MODELS_DIR, "dt_model_smotenc.pkl"))
rf_model_smotenc = safe_load_models(os.path.join(MODELS_DIR, "rf_model_smotenc.pkl"))

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
hood_le = LabelEncoder()

toodu_df["PREMISES_TYPE"] = premises_le.fit_transform(toodu_df["PREMISES_TYPE"])
toodu_df["LOCATION_TYPE"] = location_le.fit_transform(toodu_df["LOCATION_TYPE"])
toodu_df["HOOD_158"] = hood_le.fit_transform(toodu_df["HOOD_158"])
toodu_df["OFFENCE_ENCODED"] = le.fit_transform(toodu_df["OFFENCE"])

categorical_features = [
    "PREMISES_TYPE",
    "LOCATION_TYPE",
    "HOOD_158",
]

numerical_features = [
    "LONG_WGS84",
    "LAT_WGS84",
    "OCC_HOUR",
    "REPORT_HOUR",
]

label_encoders = {}

for col in categorical_features:
    label_encoders[col] = LabelEncoder()
    toodu_df[col] = label_encoders[col].fit_transform(toodu_df[col])

premises_options = create_selectbox_options(premises_le)
location_options = create_selectbox_options(location_le)
hood_options = create_selectbox_options(hood_le)

# features_filtered = toodu_df[categorical_features]`
target_filtered = toodu_df["OFFENCE_ENCODED"]

scaler = StandardScaler()
toodu_df[numerical_features] = scaler.fit_transform(toodu_df[numerical_features])
features_filtered = toodu_df[categorical_features + numerical_features]

x_train, x_test, y_train, y_test = train_test_split(features_filtered, target_filtered, test_size=0.2, random_state=42)

location_type = list(toodu_df["LOCATION_TYPE"].unique())
premises_type = list(toodu_df["PREMISES_TYPE"].unique())
hood_158 = list(toodu_df["HOOD_158"].unique())
# endregion

st.set_page_config(
    page_title="Theft Prediction Analysis",
    page_icon="📈",
    layout="centered"
)

# region Sidebar Configuration
models = {
    "Logistic Regression (SMOTENC)": lr_model_smotenc,
    "Decision Tree (SMOTENC)": dt_model_smotenc,
    "Random Forest (SMOTENC)": rf_model_smotenc,
}
model_names = list(models.keys())

st.sidebar.title("Discovery Sidebar 🌐")
st.sidebar.selectbox("Select a Model to View and Use for Predictions:", model_names, key="model")
selected_model = models[st.session_state.model]
# endregion

st.title("Theft Prediction Analysis 📈")
st.write("""
## Predictive Modeling for Theft Risk Assessment

This dashboard/page allows you to explore the various predictive and
classifications models we used to predict the likelihood of whether, 
given the parameters, a place was an `Auto Theft` or a `Non-Auto Theft`.

## How is this Different from the Other Page?

Models used in this page are trained with a different set of features,
and are trained with a different statistical technique called `SMOTENC`.
which stands for `Synthetic Minority Over-sampling Technique for
Numeric and Categorical Data`. This technique is an extension of SMOTE that
can handle both numerical and categorical data.
""")

st.write("""
On the left side where you see the sidebar, you can select the model you want to view.
""")
st.markdown("---")

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
1. `http://localhost:5000/api/v1/predict/smotec` - This is the endpoint for SMOTEC model predictions.
2. `http://localhost:5000/api/v1/summarize` - This is the endpoint for model summaries using AI.
3. `http://localhost:5000/api/v1/` - This is a general endpoint for the API.
""")

user_input_smotenc = st.session_state.get("user_inputs_smotenc", [])
entry_format = {
    # NOTE: Categorical Inputs
    "LOCATION_TYPE": st.selectbox("LOCATION_TYPE", options=location_options, format_func=lambda n: n[0]),
    "PREMISES_TYPE": st.selectbox("PREMISES_TYPE", options=premises_options, format_func=lambda n: n[0]),
    "HOOD_158": st.selectbox("HOOD_158", options=hood_options, format_func=lambda n: n[0]),

    # NOTE: Numerical Inputs
    "LONG_WGS84": st.number_input("LONG_WGS84", format="%.4f", placeholder="Longitude"),
    "LAT_WGS84": st.number_input("LAT_WGS84", format="%.4f", placeholder="Latitude"),
    "OCC_HOUR": st.number_input("OCC_HOUR", format="%.2f", placeholder="OCC Hour"),
    "REPORT_HOUR": st.number_input("REPORT_HOUR", format="%.2f", placeholder="Report Hour"),
}

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Entries", use_container_width=True):
            user_input_smotenc.clear()
            st.session_state.user_inputs_smotenc = user_input_smotenc
            st.toast("All Entries have been Cleared!", icon="✅")

    with col2:
        if st.button("Add Entry", use_container_width=True):
            user_input_smotenc.append({k: v[1] if isinstance(v, tuple) else v for k, v in entry_format.items()})
            st.session_state.user_inputs_smotenc = user_input_smotenc
            st.toast("Entry has been Successfully Added!", icon="✅")

with st.expander("Current JSON Request", expanded=True):
    st.write("Your current request looks like the following:")
    st.json(user_input_smotenc)

    st.write("""
        > **NOTE**:
        > The above JSON object is dynamically generated so long as you interact and add more entries.
        
        Once you are satisfied with the entries you have added, click on the `Predict` button below to make the
        predictions based on the entries you provided above.
    """)

toodu_df_map = pd.read_csv(os.path.join(DATA_DIR, "Theft_Over_Open_Data_Filtered.csv"))
toodu_df_map.rename(columns={"LAT_WGS84": "LAT", "LONG_WGS84": "LON"}, inplace=True)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=43.651070,
            longitude=-79.384331,
            zoom=10,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=toodu_df_map,
                get_position=["LON", "LAT"],
                get_radius=40,
                get_color=[255, 0, 0],
                pickable=True,
                auto_highlight=True,
            ),
        ],
        tooltip={
            "html": "LAT: <b>{LAT}</b><br>LON: <b>{LON}</b><br>OFFENCE: <b>{OFFENCE}</b><br>PREMISES_TYPE: <b>{"
                    "PREMISES_TYPE}</b>",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.7)",
                "color": "white",
                "padding": "10px",
                "fontSize": "14px",
                "borderRadius": "8px"
            },
        },
    )
)

if st.button("Predict Probabilities 🔮", use_container_width=True):
    st.balloons()
    st.session_state.user_inputs_smotenc = user_input_smotenc
    predictions = requests.post(
        "http://localhost:5000/api/v1/predict/smotenc",
        params={"model_name": str(selected_model.__class__.__name__)},
        json=st.session_state.user_inputs_smotenc,
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
