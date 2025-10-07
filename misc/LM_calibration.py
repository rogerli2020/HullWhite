import pandas as pd
import re
import numpy as np
from ZeroRateCurve import ExampleNSSCurve
from HullWhite import OneFactorHullWhiteModel
from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor, as_completed

# load data
df = pd.read_csv("./data/swaption_market_quotes.csv")
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
    print("Pricing swaption...")
    zcb_curve = ExampleNSSCurve()
    swaption = EuropeanSwaption(
        swaption_type=SwaptionType.PAYER,
        expiry=swap_start,
        swap_start=swap_start,
        swap_end=swap_end,
        payment_frequency=0.5,
        notional=1,
        strike=0.0,
        fixed=0.0,
    )
    swaption.set_ATM_strike_fixed_rate_and_strike(zcb_curve)
    tree = OneFactorHullWhiteTrinomialTree(hw_model, swaption.get_valuation_times(), zcb_curve, timestep)
    tree.build_tree(verbose=True)
    pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
    print("Swaption priced!")
    return pricer.price(swaption) * 10000  # convert to bps

# objective function
ITER_COUNT = 0
PREV_MAE = 0.0
def residuals(theta, dataframe, timestep=0.5, max_workers=12):
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
    market_prices = dataframe['Quoted_Premium'].values / 10000

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

    return residuals_array

# initial values!
# theta0 = [0.333333333333333333333] + [0.0020] * 3 + [0.0030] * 3 + [0.0040] * 3 + [0.0050] * 3

# Iteration: 21	 Current a: 0.010808312107682148	 Current sigma: [3.96413334e-05 9.59025699e-04 2.91565952e-02 5.81295242e-02
#  3.69920623e-02 8.99999169e-02 8.45504681e-02 4.00000000e-03
#  4.00000000e-03 5.00000000e-03 5.00000000e-03 5.00000500e-03]
# used these as starting points
theta0 = [0.010808312107682148, 3.96413334e-05, 9.59025699e-04, 2.91565952e-02, 5.81295242e-02,
          3.69920623e-02, 8.99999169e-02, 8.45504681e-02, 4.00000000e-03,
          4.00000000e-03, 5.00000000e-03, 5.00000000e-03, 5.00000500e-03]

# TRF (with very forgiving bounds...)
lower_bounds = [1e-16] + [1e-16]*12
upper_bounds = [1] + [0.09]*12
res = least_squares(residuals, theta0, args=(df,), method='trf',
                    bounds=(lower_bounds, upper_bounds), diff_step=1e-6)

# calibrated parameters!
calibrated_a = res.x[0]
calibrated_sigmas = res.x[1:]

print("Calibrated a:", calibrated_a)
print("Calibrated sigmas:", calibrated_sigmas)
print("Residual norm:", np.linalg.norm(res.fun))