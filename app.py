import streamlit as st
import pickle
import pandas as pd

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
        return pd.DataFrame(columns=df.columns)

    # Use first valid row from filtered dataset as the "anchor"
    idx = filtered_df.index[0]

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

recommendations = recommend_from_filtered(filtered_df)

# =====================
# Display Results
# =====================
if not filtered_df.empty:
    top_pick = filtered_df.iloc[0]["recommended_furniture"]
    st.subheader(f"🎯 Top Pick: **{top_pick}**")
    st.success(f"Based on your filters, we recommend starting with **{top_pick}**.")

st.subheader("📋 Recommended Options")
if recommendations.empty:
    st.warning("No recommendations found. Try adjusting your filters.")
else:
    st.dataframe(recommendations, use_container_width=True)

    # Download Button
    csv = recommendations.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Recommendations",
        data=csv,
        file_name="furniture_recommendations.csv",
        mime="text/csv",
    )
