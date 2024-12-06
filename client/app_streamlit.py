import streamlit as st

smote_models = st.Page(
    page="views/smote_models.py",
    title="SMOTE Models",
    icon=":material/cognition:",
    default=True
)

smotenc_models = st.Page(
    page="views/smotenc_models.py",
    title="SMOTENC Models",
    icon=":material/cognition_2:",
)

if __name__ == "__main__":
    nav = st.navigation({
        "SMOTE": [smote_models],
        "SMOTENC": [smotenc_models]
    })
    nav.run()

    st.sidebar.markdown("---")
    st.sidebar.success("🎉 Welcome to the Theft Prediction Analysis Dashboard!")
    st.sidebar.info("""
        In order to view the models, head over to the **SMOTE** or **SMOTENC** models page for
        more information and a detailed view of the models and their performance. You will
        also be able to interact with the models and make predictions based on the inputs
        you provide.
    """)
