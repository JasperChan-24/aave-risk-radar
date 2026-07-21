from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from aave_risk_monitor.models.pnl import (
    FixedQuoteProvider,
    LiquidatorPnLInputs,
    QuoteProvider,
    QuoteRequest,
    QuoteUnavailableError,
    QuoteValidationError,
    SwapQuote,
    ZeroXQuoteProvider,
    calculate_liquidator_pnl,
    estimate_liquidator_pnl,
)

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
TAKER = "0x1111111111111111111111111111111111111111"


def _inputs(*, minimum: bool = True) -> LiquidatorPnLInputs:
    return LiquidatorPnLInputs(
        collateral_asset=WETH,
        debt_asset=USDC,
        collateral_received_raw=10**18,
        debt_principal_raw=1_000_000_000,
        collateral_decimals=18,
        debt_decimals=6,
        flash_loan_premium_bps=5,
        gas_cost_debt_raw=10_000_000,
        taker=TAKER,
        use_minimum_swap_proceeds=minimum,
    )


def _quote() -> SwapQuote:
    return SwapQuote(
        sell_asset=WETH,
        buy_asset=USDC,
        sell_amount_raw=10**18,
        expected_buy_amount_raw=1_100_000_000,
        minimum_buy_amount_raw=1_089_000_000,
        buy_token_fee_raw=1_000_000,
        source="fixed-snapshot",
        block_number=23_000_000,
    )


def test_pnl_cost_decomposition_and_break_even() -> None:
    result = calculate_liquidator_pnl(_inputs(), _quote())

    assert result.gross_swap_proceeds_raw == 1_101_000_000
    assert result.expected_swap_proceeds_raw == 1_100_000_000
    assert result.swap_proceeds_raw == 1_089_000_000
    assert result.flash_principal_raw == 1_000_000_000
    assert result.flash_premium_raw == 500_000
    assert result.gas_cost_raw == 10_000_000
    assert result.slippage_cost_raw == 11_000_000
    assert result.quote_fee_raw == 1_000_000
    assert result.total_costs_including_principal_raw == 1_022_500_000
    assert result.net_pnl_raw == 78_500_000
    assert result.net_pnl == Decimal("78.5")
    assert result.profitable is True
    assert result.break_even_net_swap_proceeds_raw == 1_010_500_000
    assert result.break_even_gross_swap_proceeds_raw == 1_022_500_000
    assert result.break_even_debt_per_collateral == Decimal("1022.5")


def test_expected_proceeds_mode_does_not_charge_slippage_buffer() -> None:
    result = calculate_liquidator_pnl(_inputs(minimum=False), _quote())

    assert result.swap_proceeds_raw == 1_100_000_000
    assert result.slippage_cost_raw == 0
    assert result.net_pnl_raw == 89_500_000


def test_other_execution_cost_is_itemized_and_reduces_profit_one_for_one() -> None:
    baseline = calculate_liquidator_pnl(_inputs(), _quote())
    result = calculate_liquidator_pnl(
        replace(_inputs(), other_cost_debt_raw=12_000_000),
        _quote(),
    )

    assert result.other_cost_raw == 12_000_000
    assert result.total_costs_including_principal_raw == (
        baseline.total_costs_including_principal_raw + 12_000_000
    )
    assert result.net_pnl_raw == baseline.net_pnl_raw - 12_000_000
    assert result.break_even_net_swap_proceeds_raw == (
        baseline.break_even_net_swap_proceeds_raw + 12_000_000
    )


def test_quote_network_fee_is_exposed_without_double_counting_gas() -> None:
    baseline = calculate_liquidator_pnl(_inputs(), _quote())
    result = calculate_liquidator_pnl(
        _inputs(),
        replace(_quote(), network_fee_native_raw=12_345),
    )

    assert result.quote_network_fee_native_raw == 12_345
    assert result.gas_cost_raw == baseline.gas_cost_raw
    assert result.total_costs_including_principal_raw == (
        baseline.total_costs_including_principal_raw
    )
    assert result.net_pnl_raw == baseline.net_pnl_raw


def test_flash_premium_uses_v37_ceiling_rounding() -> None:
    inputs = LiquidatorPnLInputs(
        collateral_asset="COLL",
        debt_asset="DEBT",
        collateral_received_raw=1,
        debt_principal_raw=1,
        collateral_decimals=0,
        debt_decimals=0,
        flash_loan_premium_bps=1,
        gas_cost_debt_raw=0,
    )
    quote = SwapQuote(
        sell_asset="COLL",
        buy_asset="DEBT",
        sell_amount_raw=1,
        expected_buy_amount_raw=10,
    )

    result = calculate_liquidator_pnl(inputs, quote)

    assert result.flash_premium_raw == 1
    assert result.net_pnl_raw == 8


def test_fixed_quote_provider_is_deterministic_and_validates_trade() -> None:
    provider = FixedQuoteProvider(_quote())
    assert isinstance(provider, QuoteProvider)

    result = estimate_liquidator_pnl(_inputs(), provider)
    assert result.net_pnl_raw == 78_500_000

    wrong_request = QuoteRequest(
        sell_asset=WETH,
        buy_asset=USDC,
        sell_amount_raw=2 * 10**18,
        sell_decimals=18,
        buy_decimals=6,
    )
    with pytest.raises(QuoteValidationError, match="sell amount"):
        provider.get_quote(wrong_request)


def test_zerox_provider_fails_clearly_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROX_API_KEY", raising=False)
    provider = ZeroXQuoteProvider(api_key="", default_taker=TAKER)
    request = QuoteRequest(
        sell_asset=WETH,
        buy_asset=USDC,
        sell_amount_raw=10**18,
        sell_decimals=18,
        buy_decimals=6,
    )

    with pytest.raises(QuoteUnavailableError, match="ZEROX_API_KEY"):
        provider.get_quote(request)


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def _zerox_payload() -> dict[str, Any]:
    return {
        "liquidityAvailable": True,
        "sellToken": WETH.lower(),
        "buyToken": USDC.lower(),
        "sellAmount": str(10**18),
        "buyAmount": "1100000000",
        "minBuyAmount": "1089000000",
        "blockNumber": "23000000",
        "totalNetworkFee": "12345",
        "fees": {
            "integratorFee": {"amount": "1000000", "token": USDC},
            "zeroExFee": {"amount": "2000000", "token": USDC.lower()},
            "gasFee": None,
        },
        "issues": {
            "allowance": {"actual": "0", "spender": "0xabc"},
            "balance": None,
            "simulationIncomplete": False,
            "invalidSourcesPassed": [],
        },
        # Executable transaction data may be returned by 0x, but the provider
        # intentionally does not expose it through SwapQuote.
        "transaction": {"to": "0xdead", "data": "0x1234"},
    }


def test_zerox_provider_parses_read_only_quote_and_fee_breakdown() -> None:
    client = _FakeClient(_FakeResponse(_zerox_payload()))
    provider = ZeroXQuoteProvider(
        api_key="secret",
        default_taker=TAKER,
        http_client=client,
        slippage_bps=100,
    )
    request = QuoteRequest(
        sell_asset=WETH,
        buy_asset=USDC,
        sell_amount_raw=10**18,
        sell_decimals=18,
        buy_decimals=6,
        chain_id=1,
    )

    quote = provider.get_quote(request)

    assert quote.expected_buy_amount_raw == 1_100_000_000
    assert quote.minimum_buy_amount_raw == 1_089_000_000
    assert quote.buy_token_fee_raw == 3_000_000
    assert quote.block_number == 23_000_000
    assert quote.network_fee_native_raw == 12_345
    assert quote.issues == ("allowance",)
    assert quote.source == "0x-v2-read-only-firm-pricing"
    assert not hasattr(quote, "transaction")

    call = client.calls[0]
    assert call["url"].endswith("/swap/allowance-holder/quote")
    assert call["params"] == {
        "chainId": "1",
        "sellToken": WETH,
        "buyToken": USDC,
        "sellAmount": str(10**18),
        "taker": TAKER,
        "slippageBps": "100",
    }
    assert call["headers"]["0x-api-key"] == "secret"
    assert call["headers"]["0x-version"] == "v2"


def test_zerox_provider_rejects_unavailable_liquidity() -> None:
    payload = _zerox_payload()
    payload["liquidityAvailable"] = False
    provider = ZeroXQuoteProvider(
        api_key="secret",
        default_taker=TAKER,
        http_client=_FakeClient(_FakeResponse(payload)),
    )

    with pytest.raises(QuoteUnavailableError, match="no route liquidity"):
        provider.get_quote(QuoteRequest(WETH, USDC, 10**18, 18, 6))


def test_zerox_provider_surfaces_http_failure() -> None:
    provider = ZeroXQuoteProvider(
        api_key="secret",
        default_taker=TAKER,
        http_client=_FakeClient(
            _FakeResponse({}, status_code=503, error=RuntimeError("service down"))
        ),
    )

    with pytest.raises(QuoteUnavailableError, match=r"HTTP 503.*service down"):
        provider.get_quote(QuoteRequest(WETH, USDC, 10**18, 18, 6))


def test_zerox_provider_rejects_fee_in_non_buy_token() -> None:
    payload = _zerox_payload()
    payload["fees"] = {
        "integratorFee": {"amount": "1", "token": WETH},
    }
    provider = ZeroXQuoteProvider(
        api_key="secret",
        default_taker=TAKER,
        http_client=_FakeClient(_FakeResponse(payload)),
    )

    with pytest.raises(QuoteUnavailableError, match="non-buy-token fee"):
        provider.get_quote(QuoteRequest(WETH, USDC, 10**18, 18, 6))
