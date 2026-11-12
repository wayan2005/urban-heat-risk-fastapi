import pickle
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/synthetic_urban_heat_risk.csv")

features = ["month", "impervious_pct", "tree_canopy_pct", "ndvi",
            "pop_density_km2", "building_height_m", "albedo",
            "air_temp_c", "humidity_pct", "wind_speed_ms"]
X = df[features].values
y = df["heat_risk"].values   

train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    .split(X, y, groups=df["city_id"])
)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix (Low, Moderate, High):")
print(confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"]))

pickle.dump(model, open("models/heat_risk_model.pkl", "wb"))
print("\nModel saved successfully")
