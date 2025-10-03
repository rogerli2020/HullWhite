import pandas as pd
import re
import numpy as np
from ZeroRateCurve import ExampleLinearlyInterpolatedZeroRateCurve
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
    zcb_curve = ExampleLinearlyInterpolatedZeroRateCurve()
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
    tree.build_tree(verbose=False)
    pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
    return pricer.price(swaption) * 10000  # convert to bps

# objective function
def residuals(theta, dataframe, timestep=0.5, max_workers=6):
    a = theta[0]
    sigmas = theta[1:]
    print(f"Current a: {a}\t Current sigmas: {sigmas}")
    model = OneFactorHullWhiteModel(a)
    model.set_sigmas_from_vector(sigmas)

    def price_row(row):
        swap_start, swap_dur = extract_tenors(row.Description)
        swap_end = swap_start + swap_dur
        swap_start = round(swap_start, 4)
        swap_end = round(swap_end, 4)
        return price_swaption(model, swap_start, swap_end, timestep)

    prices = [0.0] * len(dataframe)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(price_row, row): i for i, row in enumerate(dataframe.itertuples())}
        for future in as_completed(futures):
            idx = futures[future]
            prices[idx] = future.result()

    return np.array(prices) - dataframe['Quoted_Premium'].values

# initial guess
theta0 = [0.01] + [0.01]*12

# Levenberg–Marquardt
# res = least_squares(residuals, theta0, args=(df.head(15),), method='lm')

# TRF
lower_bounds = [0.0] + [1e-6]*12  # a >= 0, sigmas > 0
upper_bounds = [0.1] + [0.5]*12   # set reasonable upper limits
res = least_squares(residuals, theta0, args=(df.head(25),), method='trf',
                    bounds=(lower_bounds, upper_bounds))


# calibrated parameters!
calibrated_a = res.x[0]
calibrated_sigmas = res.x[1:]

print("Calibrated a:", calibrated_a)
print("Calibrated sigmas (1y..30y):", calibrated_sigmas)
print("Residual norm:", np.linalg.norm(res.fun))
