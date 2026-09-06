import pytest

from trade_marketing_tool.options import OptionInputs, price_option, implied_volatility


def test_call_price_matches_known_reference_value():
    # Classic Hull textbook example: S=42, K=40, T=0.5, r=0.10, sigma=0.20
    # Reference Black-Scholes call price ~= 4.76
    inp = OptionInputs(spot=42, strike=40, time_to_expiry=0.5, rate=0.10, volatility=0.20)
    result = price_option(inp, "call")
    assert result.price == pytest.approx(4.76, abs=0.01)


def test_put_call_parity_holds():
    inp = OptionInputs(spot=100, strike=105, time_to_expiry=1.0, rate=0.03, volatility=0.30)
    call = price_option(inp, "call")
    put = price_option(inp, "put")
    # C - P = S*e^-qT - K*e^-rT
    import math

    lhs = call.price - put.price
    rhs = inp.spot * math.exp(-inp.dividend_yield * inp.time_to_expiry) - inp.strike * math.exp(
        -inp.rate * inp.time_to_expiry
    )
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_call_delta_is_between_0_and_1():
    inp = OptionInputs(spot=50, strike=55, time_to_expiry=0.25, rate=0.02, volatility=0.4)
    result = price_option(inp, "call")
    assert 0 <= result.delta <= 1


def test_put_delta_is_between_minus1_and_0():
    inp = OptionInputs(spot=50, strike=55, time_to_expiry=0.25, rate=0.02, volatility=0.4)
    result = price_option(inp, "put")
    assert -1 <= result.delta <= 0


def test_deep_itm_call_delta_approaches_one():
    inp = OptionInputs(spot=1000, strike=10, time_to_expiry=0.5, rate=0.02, volatility=0.2)
    result = price_option(inp, "call")
    assert result.delta > 0.99


def test_implied_volatility_recovers_the_input_volatility():
    inp = OptionInputs(spot=100, strike=100, time_to_expiry=0.5, rate=0.02, volatility=0.35)
    market_price = price_option(inp, "call").price
    recovered = implied_volatility(market_price, inp, "call")
    assert recovered == pytest.approx(0.35, abs=1e-4)


def test_invalid_option_type_raises():
    inp = OptionInputs(spot=100, strike=100, time_to_expiry=0.5, rate=0.02, volatility=0.2)
    with pytest.raises(ValueError):
        price_option(inp, "straddle")


def test_zero_time_to_expiry_raises():
    inp = OptionInputs(spot=100, strike=100, time_to_expiry=0.0, rate=0.02, volatility=0.2)
    with pytest.raises(ValueError):
        price_option(inp, "call")
