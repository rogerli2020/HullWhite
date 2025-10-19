from src.ZeroRateCurve import ExampleNSSCurve
from src.HullWhite import OneFactorHullWhiteModel
from src.Swaption import EuropeanSwaption, SwaptionType
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
import pandas as pd
import re
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor, as_completed

# use a global ZCB curve
ZCB_CURVE = ExampleNSSCurve()

# load data
df = pd.read_csv("./data/swaption_quotes_CALIBRATION.csv")
df = df.dropna()
df = df[df['Description'].str.contains("EUR Swaption Premium", na=False)]

# parse data
def extract_tenors(description):
    matches = re.findall(r'(\d+)([YM])', description.upper())
    tenors = []
    for amount, unit in matches:
        amount = float(amount)
        if unit == 'M':
            amount /= 12
        tenors.append(amount)
    
    if len(tenors) >= 2:
        return tenors[0], tenors[1]
    elif len(tenors) == 1:
        return tenors[0], None
    else:
        return None, None

# function for pricing swaptions using the models
def price_swaption(hw_model, swap_start, swap_end, timestep):
    global ZCB_CURVE
    swaption = EuropeanSwaption(
        swaption_type=SwaptionType.PAYER,
        swap_start=swap_start,
        swap_end=swap_end,
        payment_frequency=0.5,
    )
    # print(f"Pricing {swaption.__repr__()}...")
    tree = swaption.build_valuation_tree(ZCB_CURVE, set_ATM_strike=True, 
                                         model=hw_model, timestep=timestep, verbose=False)
    price = HullWhiteTreeEuropeanSwaptionPricer.price_in_bps(swaption, tree)
    print(f"Swaption {swaption.__repr__()} priced!")
    return price  # convert to bps

# objective function
ITER_COUNT = 0
PREV_MAE = 0.0
def residuals(theta, dataframe, timestep=(1/48), max_workers=12):
    global ITER_COUNT
    global PREV_MAE
    ITER_COUNT += 1

    # a = theta[0]
    # sigma = theta[1:]

    a = 0.003
    sigma = theta

    log_str = f"Iteration: {ITER_COUNT}\t Current a: {a}\t Current sigma: {sigma}"
    print(log_str)
    with open("./log.txt", "a") as f:
        f.write("\n" + log_str)

    # initialize model
    model = OneFactorHullWhiteModel(a)
    model.set_sigmas_from_vector(sigma)

    # helper function for price swaption
    def price_row(row):
        swap_start, swap_dur = extract_tenors(row.Description)
        swap_end = swap_start + swap_dur
        swap_start = round(swap_start, 4)
        swap_end = round(swap_end, 4)
        return price_swaption(model, swap_start, swap_end, timestep)

    # price them all! In parallel!
    prices = [0.0] * len(dataframe)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(price_row, row): i for i, row in enumerate(dataframe.itertuples())}
        for future in as_completed(futures):
            idx = futures[future]
            prices[idx] = future.result()

    # bps -> unscaled
    market_prices = dataframe['Quoted_Premium'].values

    # residuals
    residuals_array = np.array(prices) - market_prices

    # log and print!
    MAE = np.mean(np.abs(residuals_array))
    MAE_CHANGE = MAE - PREV_MAE
    PREV_MAE = MAE

    print(f"\tCurrent MAE: {MAE:.8f}")
    print(f"\tChange in MAE: {MAE_CHANGE:.8f}")
    with open("./log.txt", "a") as f:
        f.write(f"\n\tCurrent MAE: {MAE:.8f}")
        f.write(f"\n\tChange in MAE: {MAE_CHANGE:.8f}")

    residuals_array = (np.array(prices) - market_prices) / market_prices
    return residuals_array


# initial guess
# theta0 = [0.01] + [0.04] * 12
# theta0 = [0.02] * 12

# Calibrated a: 0.045068880765744015
# Calibrated sigmas: [0.0176809  0.01538654 0.01222778 0.06406526 0.01756987 0.03060232
#  0.02016823 0.02500033 0.01839887 0.01664046 0.03418893 0.01270629]
# theta0 = [0.003] + [
#     0.01051865, 0.00931101, 0.00999334, 0.0106191,
#     0.01073114, 0.01129566, 0.01155485, 0.01207284,
#     0.01244945, 0.01227263, 0.01200748, 0.01447145
# ]


# Iteration: 224	 Current a: 0.022698774793271204	 Current sigma: [0.01135016 0.00984644 0.00996806 0.01083984 0.01047401 0.01110636
#  0.01099313 0.01121305 0.01140323 0.01069859 0.01010777 0.01058382]
# 	Current MAE: 11.85135580
# 	Change in MAE: 0.00002476

theta0 = [0.01062891, 0.00938458, 0.00998923, 0.01065211, 0.01069531, 0.01127725,
 0.01147393, 0.01196858, 0.01229759, 0.01206488, 0.01173978, 0.01390085]

# # Levenberg-Marquardt
# res = least_squares(residuals, theta0, args=(df,), method='lm')

# TRF
lower_bounds = [1e-12] * 12
upper_bounds = [0.999999999] * 12
res = least_squares(residuals, theta0, args=(df,), method='trf',
                    bounds=(lower_bounds, upper_bounds))

# calibrated parameters!
calibrated_a = res.x[0]
calibrated_sigmas = res.x[1:]

# print("Calibrated a:", calibrated_a)
print("Calibrated sigmas:", calibrated_sigmas)
print("Residual norm:", np.linalg.norm(res.fun))