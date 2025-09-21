import joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Espera un CSV con columnas: timestamp,temp,hum,air_q,light
df = pd.read_csv("data.csv")
df = df.sort_values("timestamp")

# Features sencillas
df["air_q_ma3"] = df["air_q"].rolling(3, min_periods=1).mean()
df["air_q_next"] = df["air_q"].shift(-1)  # horizonte 1
df = df.dropna(subset=["air_q_next"])

X = df[["temp","hum","air_q","light","air_q_ma3"]].values
y = df["air_q_next"].values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
pipe.fit(X, y)

out_path = "modelo.pkl"

joblib.dump(pipe, out_path)
print(f"Guardado: {out_path}")
