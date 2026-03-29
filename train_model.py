import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# -----------------------------
# GENERATE SYNTHETIC DATA
# -----------------------------

np.random.seed(42)

data_size = 500

# Features
distance = np.random.uniform(1, 50, data_size)  # km

traffic_level = np.random.choice([1, 2, 3], data_size)  # 1=Low, 2=Medium, 3=High
vehicle_type = np.random.choice([1, 2, 3], data_size)  # 1=Bike, 2=Van, 3=Truck
priority = np.random.choice([1, 2, 3], data_size)      # 1=Low, 2=Medium, 3=High

# -----------------------------
# TARGET (ETA CALCULATION)
# -----------------------------

eta = (
    distance * 1.8 +          # distance impact
    traffic_level * 5 +       # traffic increases delay
    priority * 3 -            # priority slightly increases urgency handling
    vehicle_type * 2          # larger vehicle may be slower
)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

X = pd.DataFrame({
    "distance": distance,
    "traffic_level": traffic_level,
    "vehicle_type": vehicle_type,
    "priority": priority
})

y = eta

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# EVALUATE MODEL
# -----------------------------

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)

print(f"Model trained successfully!")
print(f"Mean Absolute Error (MAE): {round(mae, 2)} minutes")

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(model, "eta_model.pkl")

print("Model saved as eta_model.pkl")
