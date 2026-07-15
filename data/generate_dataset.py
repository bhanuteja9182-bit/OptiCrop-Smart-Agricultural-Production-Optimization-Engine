"""
generate_dataset.py
--------------------
Builds Crop_recommendation.csv — 2200 rows (100 samples x 22 crops) with
columns: N, P, K, temperature, humidity, ph, rainfall, label

Each crop is generated from agronomically realistic parameter ranges so the
overall dataset statistics closely match a real-world crop recommendation
dataset (mean N ~50.5, P ~53.4, K ~48.1, temperature ~25.6, humidity ~71.5,
ph ~6.5, rainfall ~103.5 across 2200 samples).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
SAMPLES_PER_CROP = 100

# (crop, N_mean, N_std, P_mean, P_std, K_mean, K_std,
#  temp_mean, temp_std, hum_mean, hum_std, ph_mean, ph_std, rain_mean, rain_std)
CROP_PARAMS = [
    ("rice",        80, 15, 47, 10, 40,  8, 23.7, 2.0, 82.0, 4.0, 6.4, 0.4, 236.0, 30.0),
    ("maize",       78, 15, 48, 10, 20,  5, 22.5, 3.5, 65.0, 8.0, 6.2, 0.5, 84.0, 20.0),
    ("chickpea",    40, 10, 68, 10, 80,  8, 18.9, 2.5, 16.9, 3.0, 7.3, 0.4, 80.0, 15.0),
    ("kidneybeans", 21,  6, 68, 10, 20,  5, 20.1, 2.0, 21.6, 3.0, 5.7, 0.4, 105.0, 20.0),
    ("pigeonpeas",  21,  6, 68, 10, 20,  5, 27.7, 3.5, 48.1, 10.0, 5.8, 0.5, 149.0, 30.0),
    ("mothbeans",   21,  6, 48, 10, 20,  5, 28.2, 2.5, 53.2, 10.0, 6.8, 0.5, 51.0, 15.0),
    ("mungbean",    21,  6, 47, 10, 20,  5, 28.5, 2.0, 85.5, 4.0, 6.7, 0.4, 48.0, 12.0),
    ("blackgram",   40, 10, 67, 10, 19,  5, 29.7, 2.5, 65.1, 7.0, 7.1, 0.5, 68.0, 15.0),
    ("lentil",      18,  5, 68, 10, 19,  5, 24.5, 2.5, 64.8, 8.0, 6.9, 0.4, 46.0, 12.0),
    ("pomegranate", 18,  5, 18,  5, 40,  8, 21.8, 2.5, 90.1, 3.0, 6.4, 0.4, 107.0, 15.0),
    ("banana",     100, 12, 82, 10, 50,  8, 27.4, 2.0, 80.4, 3.0, 6.0, 0.4, 104.6, 15.0),
    ("mango",       20,  6, 27,  6, 30,  6, 31.2, 2.5, 50.2, 8.0, 5.8, 0.4, 94.9, 15.0),
    ("grapes",      23,  6, 132, 8, 200, 10, 23.9, 2.0, 81.9, 3.0, 6.0, 0.3, 69.6, 10.0),
    ("watermelon",  99, 12, 17,  5, 50,  8, 25.6, 2.5, 85.2, 3.0, 6.5, 0.4, 50.8, 10.0),
    ("muskmelon",   100, 12, 17,  5, 50,  8, 28.7, 2.0, 92.3, 2.0, 6.4, 0.4, 24.7, 8.0),
    ("apple",       21,  6, 134, 8, 200, 10, 22.6, 2.0, 92.3, 2.0, 5.9, 0.3, 112.7, 15.0),
    ("orange",      19,  5, 16,  5, 10,  4, 22.8, 2.5, 92.2, 2.0, 7.0, 0.4, 110.5, 15.0),
    ("papaya",      50, 10, 59, 10, 50,  8, 33.7, 2.5, 92.4, 2.0, 6.7, 0.4, 142.6, 20.0),
    ("coconut",     22,  6, 17,  5, 31,  6, 27.4, 2.0, 94.8, 2.0, 5.9, 0.4, 175.7, 20.0),
    ("cotton",     117, 12, 46, 10, 20,  5, 23.9, 2.5, 79.8, 4.0, 6.9, 0.4, 80.1, 15.0),
    ("jute",        78, 12, 47, 10, 40,  6, 24.9, 2.0, 79.6, 3.0, 6.7, 0.4, 174.8, 20.0),
    ("coffee",      101, 12, 28,  6, 30,  6, 25.5, 2.5, 58.9, 8.0, 6.8, 0.4, 158.1, 20.0),
]

ranges = {
    "N": (0, 140), "P": (5, 145), "K": (5, 205),
    "temperature": (8.8, 43.7), "humidity": (14.3, 99.98),
    "ph": (3.5, 9.94), "rainfall": (20.2, 298.6),
}

rows = []
for (crop, n_m, n_s, p_m, p_s, k_m, k_s, t_m, t_s, h_m, h_s, ph_m, ph_s, r_m, r_s) in CROP_PARAMS:
    n = np.random.normal(n_m, n_s, SAMPLES_PER_CROP)
    p = np.random.normal(p_m, p_s, SAMPLES_PER_CROP)
    k = np.random.normal(k_m, k_s, SAMPLES_PER_CROP)
    t = np.random.normal(t_m, t_s, SAMPLES_PER_CROP)
    h = np.random.normal(h_m, h_s, SAMPLES_PER_CROP)
    ph = np.random.normal(ph_m, ph_s, SAMPLES_PER_CROP)
    r = np.random.normal(r_m, r_s, SAMPLES_PER_CROP)

    for i in range(SAMPLES_PER_CROP):
        rows.append({
            "N": round(np.clip(n[i], *ranges["N"])),
            "P": round(np.clip(p[i], *ranges["P"])),
            "K": round(np.clip(k[i], *ranges["K"])),
            "temperature": round(np.clip(t[i], *ranges["temperature"]), 6),
            "humidity": round(np.clip(h[i], *ranges["humidity"]), 6),
            "ph": round(np.clip(ph[i], *ranges["ph"]), 6),
            "rainfall": round(np.clip(r[i], *ranges["rainfall"]), 6),
            "label": crop,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv("Crop_recommendation.csv", index=False)

print(df.shape)
print(df["label"].value_counts())
print(df.describe())
