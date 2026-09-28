import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Farm Production Cost Predictor",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "farm_production_cost.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


model = load_model()
dataset = load_dataset()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌾 Farm Production Cost Prediction")
st.subheader("Machine Learning Based Agricultural Cost Prediction")

st.markdown(
    """
    Enter the details of your farm below to estimate the
    **expected total production cost** using a trained
    Machine Learning regression model.
    """
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.write("**Case Study:** 146")
st.sidebar.write("**Problem:** Farm Production Cost Prediction")
st.sidebar.write("**Task:** Regression")
st.sidebar.write("**Best Model:** Linear Regression")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts expected farm production "
    "cost based on crop, farm size, labor and other "
    "agricultural expenditure factors."
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.header("🚜 Farm Details")

col1, col2 = st.columns(2)

with col1:

    crop_types = sorted(dataset["Crop_Type"].dropna().unique().tolist())

    crop_type = st.selectbox(
        "🌱 Crop Type",
        crop_types
    )

    farm_area = st.number_input(
        "📐 Farm Area (acres)",
        min_value=0.1,
        value=5.0,
        step=0.5
    )

    seed_cost = st.number_input(
        "🌾 Seed Cost (₹)",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )

    fertilizer_usage = st.number_input(
        "🧪 Fertilizer Usage",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    labor_requirements = st.number_input(
        "👷 Labor Requirements",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


with col2:

    irrigation_cost = st.number_input(
        "💧 Irrigation Cost (₹)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    pesticide_usage = st.number_input(
        "🧴 Pesticide Usage",
        min_value=0.0,
        value=20.0,
        step=5.0
    )

    machinery_cost = st.number_input(
        "🚜 Machinery Cost (₹)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    transportation_cost = st.number_input(
        "🚚 Transportation Cost (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown("---")

if st.button("🔮 Predict Production Cost", type="primary"):

    input_data = pd.DataFrame({
        "Crop_Type": [crop_type],
        "Farm_Area": [farm_area],
        "Seed_Cost": [seed_cost],
        "Fertilizer_Usage": [fertilizer_usage],
        "Labor_Requirements": [labor_requirements],
        "Irrigation_Cost": [irrigation_cost],
        "Pesticide_Usage": [pesticide_usage],
        "Machinery_Cost": [machinery_cost],
        "Transportation_Cost": [transportation_cost]
    })

    try:

        prediction = model.predict(input_data)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            label="💰 Expected Total Production Cost",
            value=f"₹ {prediction:,.2f}"
        )

        st.markdown("### 📋 Input Summary")

        display_data = input_data.T.reset_index()
        display_data.columns = ["Parameter", "Value"]
        # Cast Value to str to avoid PyArrow mixed-type serialization error
        display_data["Value"] = display_data["Value"].astype(str)

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True,
        )

    except Exception as e:

        st.error("Unable to generate prediction.")

        st.exception(e)


# --------------------------------------------------
# PROJECT DESCRIPTION
# --------------------------------------------------
st.markdown("---")

st.header("📊 About the Model")

st.write(
    """
    This project compares five Machine Learning regression algorithms:
    """
)

models = [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
    "Gradient Boosting Regression"
]

for i, model_name in enumerate(models, 1):
    st.write(f"{i}. {model_name}")


st.markdown("---")

st.caption(
    "ITM Skills University | Machine Learning Fundamentals | Case Study 146"
)