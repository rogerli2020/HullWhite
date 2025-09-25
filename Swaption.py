from enum import Enum

class SwaptionType(Enum):
    PAYER = "payer"
    RECEIVER = "receiver"

class SettlementType(Enum):
    PHYSICAL = "physical"
    CASH = "cash"

class Swaption:
    def __init__(self, swaption_type: SwaptionType,
                 expiry: float, swap_start: float, swap_end: float,
                 payment_frequency: float, notional: float, strike: float) -> None:
        self.swaption_type = swaption_type
        self.expiry = expiry
        self.swap_start = swap_start
        self.swap_end = swap_end
        self.payment_frequency = payment_frequency
        self.notional = notional
        self.strike = strike
