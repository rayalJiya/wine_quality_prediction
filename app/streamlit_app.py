import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_utils import FEATURES  # noqa: E402
from predictor import (  # noqa: E402
    load_meta, out_of_range_features, predict_quality,
)

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Prediction System")
st.write("Enter the chemical properties of a wine to predict its quality score.")

try:
    meta = load_meta()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

LABELS = {
    "fixed acidity": "Fixed Acidity",
    "volatile acidity": "Volatile Acidity",
    "citric acid": "Citric Acid",
    "residual sugar": "Residual Sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free Sulfur Dioxide",
    "total sulfur dioxide": "Total Sulfur Dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol (%)",
}
STEPS = {"density": 0.0001, "chlorides": 0.001, "citric acid": 0.01,
         "volatile acidity": 0.01, "sulphates": 0.01, "pH": 0.01}
FORMATS = {"density": "%.5f", "chlorides": "%.3f", "citric acid": "%.2f",
           "volatile acidity": "%.3f", "sulphates": "%.2f", "pH": "%.2f"}

with st.form("wine_form"):
    values = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        m = meta[f]
        with cols[i % 2]:
            values[f] = st.number_input(
                LABELS[f],
                min_value=0.0,
                value=float(m["median"]),
                step=STEPS.get(f, 0.1),
                format=FORMATS.get(f, "%.2f"),
                help=f"Training data range: {m['min']:g} to {m['max']:g}",
            )
    submitted = st.form_submit_button("Predict Quality")

if submitted:
    try:
        result = predict_quality(values)
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Wine Quality", result["predicted_quality"])
    c2.metric("Quality Category", result["category"])
    c3.metric("Prediction Confidence", f"{result['confidence'] * 100:.0f}%")

    if result["confidence"] < 0.5:
        st.warning("Low confidence: the model is not sure about this prediction. "
                   "Neighbouring quality scores are also likely.")

    outside = out_of_range_features(values)
    if outside:
        st.warning("These values are outside the range seen during training, "
                   "so the prediction may be less reliable: "
                   + ", ".join(LABELS[f] for f in outside))

    st.subheader("Probability of each quality score")
    prob_df = pd.DataFrame({
        "Quality": [str(k) for k in result["probabilities"]],
        "Probability (%)": [round(v * 100, 1) for v in result["probabilities"].values()],
    }).set_index("Quality")
    st.bar_chart(prob_df)
    st.dataframe(prob_df.T, use_container_width=True)

st.caption(
    "Categories: 3-4 = Poor, 5 = Average, 6 = Good, 7-8 = Excellent. "
    "Confidence is the model's predicted probability for the chosen class, "
    "not a guarantee. Predictions are based only on chemical properties."
)