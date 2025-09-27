from enum import Enum

class SwaptionType(Enum):
    PAYER = "payer"
    RECEIVER = "receiver"

class EuropeanSwaption:
    def __init__(self, swaption_type: SwaptionType,
                 expiry: float, swap_start: float, swap_end: float,
                 payment_frequency: float, notional: float, strike: float, 
                 fixed: float, spread: float) -> None:
        self.swaption_type = swaption_type
        self.expiry = expiry
        self.swap_start = swap_start
        self.swap_end = swap_end
        self.payment_frequency = payment_frequency
        self.notional = notional
        self.strike = strike
        self.fixed = fixed
        self.spread = spread
    
    def get_valuation_times(self) -> list[float]:
        valuation_times = [0.0]

        num_payments = int(round((self.swap_end - self.swap_start) / self.payment_frequency))
        for i in range(num_payments + 1):
            t = self.swap_start + i * self.payment_frequency
            t = min(t, self.swap_end)
            valuation_times.append(t)

        return valuation_times