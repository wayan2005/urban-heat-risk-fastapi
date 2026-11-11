import streamlit as st
import requests
from config import API_URL

st.set_page_config(page_title="Urban Heat Risk Prediction", page_icon="🌡️")
st.title("🌡️ Urban Heat Risk Prediction")
st.write("Enter the neighbourhood details to predict its heat-risk level.")

month = st.slider("Month", 1, 12, 7)
impervious_pct = st.slider("Impervious surface (%)", 0, 100, 80)
tree_canopy_pct = st.slider("Tree canopy (%)", 0, 70, 10)
ndvi = st.slider("NDVI (vegetation index)", 0.0, 0.8, 0.2, 0.01)
pop_density_km2 = st.number_input("Population density (per km²)", 200, 60000, 12000, 500)
building_height_m = st.slider("Average building height (m)", 3, 60, 20)
albedo = st.slider("Surface albedo (reflectivity)", 0.08, 0.35, 0.12, 0.01)
air_temp_c = st.slider("Air temperature (°C)", -5, 45, 35)
humidity_pct = st.slider("Humidity (%)", 15, 95, 50)
wind_speed_ms = st.slider("Wind speed (m/s)", 0.3, 12.0, 2.0, 0.1)

if st.button("Predict Heat Risk"):
    payload = {
        "month": month, "impervious_pct": impervious_pct,
        "tree_canopy_pct": tree_canopy_pct, "ndvi": ndvi,
        "pop_density_km2": pop_density_km2, "building_height_m": building_height_m,
        "albedo": albedo, "air_temp_c": air_temp_c,
        "humidity_pct": humidity_pct, "wind_speed_ms": wind_speed_ms,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}: {e}")
        st.stop()

    result, probs = out["heat_risk"], out["probabilities"]
    if result == "Low":
        st.success("Heat Risk: LOW 🟢")
    elif result == "Moderate":
        st.warning("Heat Risk: MODERATE 🟡")
    else:
        st.error("Heat Risk: HIGH 🔴")
    st.write({k: f"{v:.0%}" for k, v in probs.items()})

st.caption("Student project trained on a SYNTHETIC dataset (not real measurements).")
