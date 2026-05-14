import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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

@st.cache_data
def train_model(df):
    model_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]

    model_df = df[model_features + ["median_house_value"]].dropna()

    X = model_df[model_features]
    y = model_df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2, model_features


model, mae, r2, model_features = train_model(df)

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

st.subheader("Machine Learning Model Performance")

ml_col1, ml_col2 = st.columns(2)

ml_col1.metric("Model MAE", f"${mae:,.0f}")
ml_col2.metric("Model R² Score", f"{r2:.3f}")

st.caption(
    "The linear regression model predicts median house value using geographic, housing, population, and income features."
)

st.subheader("House Value Prediction")

st.markdown("Enter housing features below to predict the median house value.")

input_col1, input_col2, input_col3, input_col4 = st.columns(4)

with input_col1:
    input_longitude = st.number_input("Longitude", value=-118.0)
    input_latitude = st.number_input("Latitude", value=34.0)

with input_col2:
    input_age = st.number_input("Housing Median Age", value=30.0)
    input_rooms = st.number_input("Total Rooms", value=2500.0)

with input_col3:
    input_bedrooms = st.number_input("Total Bedrooms", value=500.0)
    input_population = st.number_input("Population", value=1200.0)

with input_col4:
    input_households = st.number_input("Households", value=450.0)
    input_income = st.number_input("Median Income", value=4.0)

input_data = pd.DataFrame(
    [[
        input_longitude,
        input_latitude,
        input_age,
        input_rooms,
        input_bedrooms,
        input_population,
        input_households,
        input_income
    ]],
    columns=model_features
)

predicted_value = model.predict(input_data)[0]

st.metric("Predicted Median House Value", f"${predicted_value:,.0f}")

st.subheader("Geographic Distribution")

if not filtered_df.empty:
    st.map(filtered_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
else:
    st.warning("No data available for the selected filters.")

st.subheader("Distribution of Median House Value")

chart_col, _ = st.columns([0.75, 0.25])

with chart_col:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(filtered_df["median_house_value"], bins=30)
    ax.set_xlabel("Median House Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Median House Value Distribution")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)


st.subheader("Relationship Between Income and House Value")

chart_col, _ = st.columns([0.75, 0.25])

with chart_col:
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    ax2.scatter(
        filtered_df["median_income"],
        filtered_df["median_house_value"],
        alpha=0.35,
        s=12
    )

    ax2.set_xlabel("Median Income")
    ax2.set_ylabel("Median House Value")
    ax2.set_title("Median Income vs. Median House Value")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=False)


st.subheader("Average House Value by Ocean Proximity")

grouped_price = (
    filtered_df
    .groupby("ocean_proximity")["median_house_value"]
    .mean()
    .sort_values(ascending=False)
)

chart_col, _ = st.columns([0.75, 0.25])

with chart_col:
    fig3, ax3 = plt.subplots(figsize=(7, 3.8))

    grouped_price.plot(kind="bar", ax=ax3)

    ax3.set_xlabel("Ocean Proximity")
    ax3.set_ylabel("Average Median House Value")
    ax3.set_title("Average House Value by Ocean Proximity")
    ax3.tick_params(axis="x", rotation=30)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=False)


st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["float64", "int64"])
corr_matrix = numeric_df.corr()

chart_col, _ = st.columns([0.82, 0.18])

with chart_col:
    fig4, ax4 = plt.subplots(figsize=(8, 6))

    im = ax4.imshow(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))

    ax4.set_xticklabels(
        corr_matrix.columns,
        rotation=45,
        ha="right",
        fontsize=9
    )
    ax4.set_yticklabels(
        corr_matrix.columns,
        fontsize=9
    )

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            ax4.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=7
            )

    ax4.set_title("Correlation Heatmap", fontsize=13, pad=12)

    cbar = fig4.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label("Correlation Coefficient")

    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=False)


st.subheader("Filtered Data Preview")

st.dataframe(filtered_df.head(100), use_container_width=True)
