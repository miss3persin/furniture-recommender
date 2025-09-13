import streamlit as st
import pickle
import pandas as pd

# Load model artifacts
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
df = artifacts["df"]
similarity_matrix = artifacts["similarity"]

st.title("🛋️ Furniture Recommendation System")
st.write("Powered by **Metaflow Pipeline + Streamlit**")

# Sidebar input: choose based on "recommended_furniture" instead of non-existent "furniture_item"
furniture_choice = st.sidebar.selectbox(
    "Pick a furniture item:",
    df["recommended_furniture"].unique()
)

# Recommendation logic
def recommend(item, top_n=5):
    idx = df[df["recommended_furniture"] == item].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:top_n+1]]
    return df.iloc[top_indices][[
        "apartment_type",
        "location",
        "budget_range",
        "preferred_style",
        "recommended_furniture"
    ]]

st.subheader(f"🎯 Recommendations for: {furniture_choice}")
results = recommend(furniture_choice)
st.dataframe(results)
