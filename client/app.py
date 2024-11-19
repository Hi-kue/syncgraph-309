import os
import sys
from constants import OPENROUTER_API_URL, OPENROUTER_API_KEY

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI, OpenAIError

client = OpenAI(
    base_url=OPENROUTER_API_URL,
    api_key=OPENROUTER_API_KEY,
)

# Introduction Section
st.write("# Theft Over Open Data - Analysis and Visualization")


# Data Exploration Section with Streamlit

# Geographical Visualization and Mapping Section

# LogisticRegression Model & Tree Visualization Section

# Model Evaluation Section

# AI Evaluation and Explanation Section
@st.cache
async def get_explanation() -> str:
    response = await client.chat.completions.create(
        model="google/gemini-flash-1.5-exp",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "<enter a descriptive user msg here>"
            },
        ]
    )
    return response.choices[0].message.content


def run_app() -> None:
    os.system("streamlit run app.py --server.port 8501")


if __name__ == "__main__":
    run_app()
