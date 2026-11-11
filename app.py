import pickle
from config import MODEL_PATH

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_heat_risk(month, impervious_pct, tree_canopy_pct, ndvi,
                      pop_density_km2, building_height_m, albedo,
                      air_temp_c, humidity_pct, wind_speed_ms):
    data = [[month, impervious_pct, tree_canopy_pct, ndvi,
             pop_density_km2, building_height_m, albedo,
             air_temp_c, humidity_pct, wind_speed_ms]]

    prediction = model.predict(data)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(data)[0]))
    return prediction, probabilities
