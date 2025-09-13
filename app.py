import streamlit as st
import pickle
import pandas as pd
import random

# =====================
# Load Model Artifacts
# =====================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
df = artifacts["df"]
similarity_matrix = artifacts["similarity"]

st.set_page_config(page_title="Furniture Recommendation System", layout="wide")
st.title("🛋️ Smart Furniture Recommendation System")
st.caption("Powered by **Metaflow + Streamlit**")

# =====================
# Sidebar Filters
# =====================
st.sidebar.header("🔧 Filters")

apartment_type = st.sidebar.selectbox(
    "Choose apartment type",
    options=["Any"] + sorted(df["apartment_type"].unique().tolist())
)

location = st.sidebar.selectbox(
    "Choose location",
    options=["Any"] + sorted(df["location"].unique().tolist())
)

budget = st.sidebar.selectbox(
    "Choose budget range",
    options=["Any"] + sorted(df["budget_range"].unique().tolist())
)

style = st.sidebar.selectbox(
    "Preferred style",
    options=["Any"] + sorted(df["preferred_style"].unique().tolist())
)

search_query = st.sidebar.text_input("🔍 Search furniture by name")

# =====================
# Filter Dataset
# =====================
filtered_df = df.copy()

if apartment_type != "Any":
    filtered_df = filtered_df[filtered_df["apartment_type"] == apartment_type]

if location != "Any":
    filtered_df = filtered_df[filtered_df["location"] == location]

if budget != "Any":
    filtered_df = filtered_df[filtered_df["budget_range"] == budget]

if style != "Any":
    filtered_df = filtered_df[filtered_df["preferred_style"] == style]

if search_query.strip():
    filtered_df = filtered_df[
        filtered_df["recommended_furniture"].str.contains(search_query, case=False, na=False)
    ]

# =====================
# Recommendation Logic
# =====================
def recommend_from_filtered(filtered_df, top_n=5):
    if filtered_df.empty:
        return []

    idx = filtered_df.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[1:top_n+1]]
    recs = df.iloc[top_indices]["recommended_furniture"].drop_duplicates().tolist()
    return recs

recommendations = recommend_from_filtered(filtered_df)

# =====================
# Display Results
# =====================
if not filtered_df.empty:
    top_pick = filtered_df.iloc[0]["recommended_furniture"]
    st.subheader(f"🎯 Top Pick: **{top_pick}**")

st.subheader("📋 Recommended Options")
if not recommendations:
    st.warning("No recommendations found. Try adjusting your filters.")
else:
    for i, item in enumerate(recommendations, start=1):
        with st.container():
            st.markdown(
                f"""
                <div style="
                    padding:15px; 
                    border-radius:12px; 
                    background-color:#ffffff; 
                    border:1px solid #ddd;
                    margin-bottom:12px; 
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);
                ">
                    <h4 style="margin:0; color:#333;">{item}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================
# Surprise Me Feature
# =====================
st.subheader("🎲 Feeling Lucky?")
if st.button("Surprise Me"):
    random_item = df.sample(1).iloc[0]["recommended_furniture"]
    st.info(f"✨ Surprise Pick: **{random_item}**")


