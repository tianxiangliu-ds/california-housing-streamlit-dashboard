import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="California Housing Dashboard",
    page_icon="🏠",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df


df = load_data()

st.title("🏠 California Housing Streamlit Dashboard")
st.markdown(
    "This dashboard explores the California Housing dataset with interactive filters, "
    "geographic visualization, and housing value distribution analysis."
)

st.sidebar.header("Filters")

price_min = int(df["median_house_value"].min())
price_max = int(df["median_house_value"].max())

price_range = st.sidebar.slider(
    "Median House Value Range",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max)
)

location_type = st.sidebar.multiselect(
    "Ocean Proximity",
    options=sorted(df["ocean_proximity"].unique()),
    default=sorted(df["ocean_proximity"].unique())
)

income_level = st.sidebar.radio(
    "Income Level",
    ["All", "Low", "Medium", "High"]
)

filtered_df = df[
    (df["median_house_value"] >= price_range[0]) &
    (df["median_house_value"] <= price_range[1]) &
    (df["ocean_proximity"].isin(location_type))
].copy()

if income_level == "Low":
    filtered_df = filtered_df[filtered_df["median_income"] <= 2.5]
elif income_level == "Medium":
    filtered_df = filtered_df[
        (filtered_df["median_income"] > 2.5) &
        (filtered_df["median_income"] < 4.5)
    ]
elif income_level == "High":
    filtered_df = filtered_df[filtered_df["median_income"] >= 4.5]

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Records", len(filtered_df))
col2.metric("Average House Value", f"${filtered_df['median_house_value'].mean():,.0f}")
col3.metric("Average Median Income", f"{filtered_df['median_income'].mean():.2f}")

st.subheader("Geographic Distribution")

if not filtered_df.empty:
    st.map(filtered_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
else:
    st.warning("No data available for the selected filters.")

st.subheader("Distribution of Median House Value")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(filtered_df["median_house_value"], bins=30)
ax.set_xlabel("Median House Value")
ax.set_ylabel("Frequency")
ax.set_title("Median House Value Distribution")
st.pyplot(fig)

st.subheader("Filtered Data Preview")

st.dataframe(filtered_df.head(100))