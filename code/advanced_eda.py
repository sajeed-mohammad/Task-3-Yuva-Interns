import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/logistics_data.csv")

print("Shape:", df.shape)
print(df.describe(include="all"))

delay_rate = df["Delivery_Delay"].mean() * 100
print(f"Delay Rate: {delay_rate:.2f}%")
print(f"On-Time Rate: {100-delay_rate:.2f}%")

for col in ["Transportation_Mode", "Traffic_Level", "Weather_Condition", "Warehouse"]:
    print(f"\nDelay rate by {col}:")
    print((df.groupby(col)["Delivery_Delay"].mean() * 100).round(2))

# Central tendency
print("\nMedian distance:", df["Delivery_Distance_km"].median())
print("Mean actual delivery time:", df["Actual_Delivery_Time_hr"].mean())
print("Median processing time:", df["Warehouse_Processing_Time_min"].median())

# Correlations
print("\nNumeric correlations:")
print(df.select_dtypes("number").corr()["Delivery_Delay"].sort_values(ascending=False))

# Visualizations
df["Delivery_Delay"].value_counts().sort_index().plot(kind="bar")
plt.title("Delivery Delay Distribution")
plt.tight_layout()
plt.savefig("../visualizations/delay_distribution.png", dpi=150)
plt.close()

traffic = df.groupby("Traffic_Level")["Delivery_Delay"].mean() * 100
traffic.plot(kind="bar")
plt.title("Delay Rate by Traffic Level")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()
plt.savefig("../visualizations/traffic_impact.png", dpi=150)
plt.close()

plt.scatter(df["Delivery_Distance_km"], df["Actual_Delivery_Time_hr"], s=18, alpha=0.6)
plt.xlabel("Delivery Distance (km)")
plt.ylabel("Actual Delivery Time (hours)")
plt.title("Distance vs Actual Delivery Time")
plt.tight_layout()
plt.savefig("../visualizations/distance_vs_delivery.png", dpi=150)
plt.close()

corr = df.select_dtypes("number").corr()
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("../visualizations/correlation_heatmap.png", dpi=150)
plt.close()
