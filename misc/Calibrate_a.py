import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("./data/euro_short_rates.csv")

# Parse dates and sort
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")

# short rate column
rate_col = "Euro short-term rate - Volume-weighted trimmed mean rate (EST.B.EU000A2X2A25.WT)"
df["r_t"] = df[rate_col].astype(float)

# throw out outliers most likely due to gov't intervention
df["delta_r"] = df["r_t"].diff()
Q1 = df["delta_r"].quantile(0.25)
Q3 = df["delta_r"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df["delta_r"] >= lower_bound) & (df["delta_r"] <= upper_bound)].copy()

# assume AR(1) dynamcs
df_clean["r_t_minus1"] = df_clean["r_t"].shift(1)
df_clean = df_clean.dropna()
X = sm.add_constant(df_clean["r_t_minus1"])
y = df_clean["r_t"]

# REGRESS!
model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["r_t_minus1"]

# annualize!
delta_t = 1/252

# mean reversion kappa
kappa = -np.log(beta) / delta_t

print(model.summary())
print("-----------------------------------------")
print(f"Estimated beta = {beta:.6f}")
print(f"Estimated kappa = {kappa:.6f} per year")


# -----------------------------------------
# Estimated beta = 0.999402
# Estimated kappa = 0.150828 per year