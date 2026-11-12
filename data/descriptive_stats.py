import pandas as pd

df = pd.read_csv("data/synthetic_urban_heat_risk.csv")

features = [
    "month",
    "ndvi",
    "impervious_pct",
    "tree_canopy_pct",
    "pop_density_km2",
    "building_height_m",
    "albedo",
    "air_temp_c",
    "humidity_pct",
    "wind_speed_ms"
]

stats = df[features].describe().T

stats = stats[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
].round(2)

print(stats.to_string())

stats.to_csv("descriptive_statistics.csv")