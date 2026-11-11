import pickle
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("data/synthetic_urban_heat_risk.csv")

# 2. Choose input columns (X) and the column we want to predict (y)
features = ["month", "impervious_pct", "tree_canopy_pct", "ndvi",
            "pop_density_km2", "building_height_m", "albedo",
            "air_temp_c", "humidity_pct", "wind_speed_ms"]
X = df[features].values
y = df["heat_risk"].values   # Low / Moderate / High

# 3. Split into train and test data
#    (split by city, so a city used for training is never used for testing)
train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    .split(X, y, groups=df["city_id"])
)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 4. Train the model
#    Logistic Regression needs the inputs scaled first, so we chain
#    StandardScaler + LogisticRegression into one model.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix (Low, Moderate, High):")
print(confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"]))

# 6. Save the model
pickle.dump(model, open("models/heat_risk_model.pkl", "wb"))
print("\nModel saved successfully")
