import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/synthetic_urban_heat_risk.csv")

# Count each heat-risk category
counts = df["heat_risk"].value_counts().reindex(
    ["Low", "Moderate", "High"]
)

print("Heat Risk Class Distribution:")
print(counts)

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(counts.index, counts.values)

plt.title("Class Distribution of Urban Heat Risk Dataset")
plt.xlabel("Heat Risk Category")
plt.ylabel("Number of Records")

# Display values above bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        int(bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save chart
plt.savefig("heat_risk_distribution.png", dpi=300)

plt.show()