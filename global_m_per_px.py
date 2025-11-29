import pandas as pd
import numpy as np
import math

# spikeball geometry
BALL_CIRC_M = 0.3048
BALL_R_M = BALL_CIRC_M / (2 * math.pi)

df = pd.read_csv("outputs/Shallow/centers/shallow1_centers.csv")

r = pd.to_numeric(df["radius_px"], errors="coerce")
valid = r[~np.isnan(r)]
r_mean = valid.mean()

m_per_px_global = BALL_R_M / r_mean
print("mean pixel radius:", r_mean)
print("global m_per_px:", m_per_px_global)