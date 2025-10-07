import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import pandas as pd
from matplotlib.pyplot import plot

t: np.array
y: np.array

df: pd.DataFrame = pd.read_csv("./data/zero_rates.csv")

t, y = df["t"], df["rate"]
t, y = np.array(t), np.array(y)

curve, status = calibrate_ns_ols(t, y, tau0=1.0)
assert status.success
print(curve)

y = curve
t = np.linspace(0, 30, 100)
plot(t, y(t))

print(y(1.5))