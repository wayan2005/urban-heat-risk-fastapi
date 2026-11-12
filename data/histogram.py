import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data\synthetic_urban_heat_risk.csv")

# Select model features
features = [
    "month", "ndvi", "impervious_pct", "tree_canopy_pct",
    "pop_density_km2", "building_height_m", "albedo",
    "air_temp_c", "humidity_pct", "wind_speed_ms"
]

# Plot distributions
df[features].hist(figsize=(14, 10), bins=20)

plt.suptitle("Distribution of Environmental Features")
plt.tight_layout()

plt.savefig("feature_distributions.png", dpi=300)
plt.show()