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
    print(f"Pricing {swaption.__repr__()}...")
    tree = swaption.build_valuation_tree(ZCB_CURVE, set_ATM_strike=True, 
                                         model=hw_model, timestep=timestep, verbose=True)
    price = HullWhiteTreeEuropeanSwaptionPricer.price_in_bps(swaption, tree)
    print(f"Swaption {swaption.__repr__()} priced!")
    return price  # convert to bps

# objective function
ITER_COUNT = 0
PREV_MAE = 0.0
def residuals(theta, dataframe, timestep=0.125, max_workers=12):
    global ITER_COUNT
    global PREV_MAE
    ITER_COUNT += 1

    a = theta[0]
    sigma = theta[1:]

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

# Calibrated a: 0.04306833078825948
# Calibrated sigmas: [0.00794198 0.01521866 0.01240818 0.02172224 0.014714   0.01666154
#  0.01499591 0.01400028 0.01388566 0.01224445 0.01122142 0.00899092]
theta0 = [0.04306833078825948] + [0.00794198, 0.01521866, 0.01240818, 0.02172224, 0.014714, 0.01666154,
                                 0.01499591, 0.01400028, 0.01388566, 0.01224445, 0.01122142, 0.00899092]

# # Levenberg-Marquardt
# res = least_squares(residuals, theta0, args=(df,), method='lm')

# TRF
lower_bounds = [1e-12] * 13
upper_bounds = [0.99999] * 13
res = least_squares(residuals, theta0, args=(df,), method='trf',
                    bounds=(lower_bounds, upper_bounds))

# calibrated parameters!
calibrated_a = res.x[0]
calibrated_sigmas = res.x[1:]

print("Calibrated a:", calibrated_a)
print("Calibrated sigmas:", calibrated_sigmas)
print("Residual norm:", np.linalg.norm(res.fun))