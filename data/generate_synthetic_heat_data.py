"""
SYNTHETIC urban heat-risk dataset (NOT real measurements).
Every row is one neighbourhood in one month. The label `heat_risk` is derived
from a heat-anomaly score built from physically plausible relationships
(more pavement/density/air temp -> hotter; more trees/vegetation/wind/water -> cooler)
plus city-level and random noise. Use it to practise classification only.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

ZONES = {"Af":(2,27),"Aw":(12,27),"BWh":(26,26),"BSh":(24,24),"Cfa":(35,16),
         "Cfb":(50,10),"Csa":(38,16),"Dfb":(52,5),"Dwa":(40,10)}   # (typical |lat|, annual mean temp)
rows = []
for cid in range(1, 61):
    z = rng.choice(list(ZONES)); lat0, mat = ZONES[z]
    lat = (lat0 + rng.normal(0, 3)) * rng.choice([1, -1], p=[.75, .25])
    elev = max(0, rng.gamma(2, 120)); coast = rng.exponential(150)
    city_eff = rng.normal(0, 0.35)                       # hidden city-level effect
    for nid in range(1, 11):
        imp = np.clip(rng.normal(60, 20), 5, 98)
        canopy = np.clip(rng.normal(30 - 0.25*(imp-60), 9), 1, 70)
        pop = np.clip(rng.lognormal(np.log(1500 + 120*imp), 0.5), 200, 60000)
        height = np.clip(rng.normal(6 + 0.12*imp + pop/6000, 3), 3, 60)
        albedo = np.clip(rng.normal(0.18 - 0.0005*(imp-60), 0.03), 0.08, 0.35)
        water = rng.exponential(2.5)                     # km to nearest water body
        for m in range(1, 13):
            season = np.cos(2*np.pi*(m-7)/12) * (1 if lat >= 0 else -1)   # +1 = peak summer
            air = mat - 0.0065*elev + (abs(lat)/12 + 6)*season*0.7 + rng.normal(0, 1.2)
            ndvi = np.clip(0.18 + 0.004*canopy + 0.07*season*(abs(lat) > 20) + rng.normal(0, .03), 0.02, 0.8)
            hum = np.clip(rng.normal(60 - 0.8*(air-20) + (15 if z in ("Af","Cfa") else 0), 10), 15, 95)
            wind = np.clip(rng.gamma(3, 1.0) - 0.02*height, 0.3, 12)
            anomaly = (0.035*imp - 0.040*canopy - 3.0*ndvi + 0.00004*pop + 0.020*height
                       - 6.0*(albedo-0.18) + 0.11*(air-15) - 0.12*wind - 0.04*water
                       + 0.01*(hum-50)*(air > 28) + city_eff + rng.normal(0, 0.55) + 2.2)
            rows.append(dict(city_id=f"C{cid:02d}", neighbourhood_id=f"C{cid:02d}_N{nid:02d}", month=m,
                climate_zone=z, latitude=round(lat,2), elevation_m=round(elev), coast_dist_km=round(coast,1),
                impervious_pct=round(imp,1), tree_canopy_pct=round(canopy,1), ndvi=round(ndvi,3),
                pop_density_km2=round(pop), building_height_m=round(height,1), albedo=round(albedo,3),
                water_dist_km=round(water,2), air_temp_c=round(air,1), humidity_pct=round(hum,1),
                wind_speed_ms=round(wind,1), heat_anomaly_c=round(anomaly,2)))
df = pd.DataFrame(rows)
lo, hi = df.heat_anomaly_c.quantile([.35, .75])
lo, hi = round(lo, 1), round(hi, 1)
df["heat_risk"] = np.where(df.heat_anomaly_c < lo, "Low", np.where(df.heat_anomaly_c < hi, "Moderate", "High"))
print(f"thresholds: Low < {lo} <= Moderate < {hi} <= High")
print(df.heat_risk.value_counts())
df.to_csv("synthetic_urban_heat_risk.csv", index=False)
