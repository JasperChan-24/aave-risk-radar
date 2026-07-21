"""Liquidator net-P&L model with pluggable, read-only swap quotes.

The pure calculation consumes a :class:`SwapQuote`; quote acquisition is kept
behind :class:`QuoteProvider`. Every monetary output is denominated in raw debt
token units. Providers must therefore normalize explicit quote fees into the
buy/debt token before returning a quote.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from .liquidation import (
    PERCENTAGE_FACTOR,
    LiquidationResult,
    percent_mul_ceil,
    token_unit,
)


class PnLError(ValueError):
    """Base error for invalid P&L inputs or quote/model mismatches."""


class QuoteError(PnLError):
    """Base error for quote providers."""


class QuoteUnavailableError(QuoteError):
    """Raised when a provider cannot supply a trustworthy quote."""


class QuoteValidationError(QuoteError):
    """Raised when a quote does not correspond to the requested trade."""


def _require_raw(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in raw token units")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise PnLError(f"{name} must be {qualifier}")


def _require_asset(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PnLError(f"{name} must be a non-empty token identifier")


def raw_to_decimal(amount_raw: int, decimals: int) -> Decimal:
    """Convert an integer token amount to an exact display Decimal."""

    _require_raw("amount_raw", amount_raw)
    return Decimal(amount_raw) / Decimal(token_unit(decimals))


@dataclass(frozen=True, slots=True)
class QuoteRequest:
    """A request to sell seized collateral for the repaid debt asset."""

    sell_asset: str
    buy_asset: str
    sell_amount_raw: int
    sell_decimals: int
    buy_decimals: int
    chain_id: int = 1
    taker: str | None = None

    def __post_init__(self) -> None:
        _require_asset("sell_asset", self.sell_asset)
        _require_asset("buy_asset", self.buy_asset)
        _require_raw("sell_amount_raw", self.sell_amount_raw, positive=True)
        token_unit(self.sell_decimals)
        token_unit(self.buy_decimals)
        _require_raw("chain_id", self.chain_id, positive=True)
        if self.taker is not None:
            _require_asset("taker", self.taker)


@dataclass(frozen=True, slots=True)
class SwapQuote:
    """Normalized quote in raw buy/debt token units.

    ``expected_buy_amount_raw`` is the amount expected after the quote's
    explicit fees. ``minimum_buy_amount_raw`` is the provider's slippage-safe
    minimum. ``buy_token_fee_raw`` is informational but lets the P&L rebuild a
    pre-fee gross amount without charging the fee twice.
    """

    sell_asset: str
    buy_asset: str
    sell_amount_raw: int
    expected_buy_amount_raw: int
    minimum_buy_amount_raw: int | None = None
    buy_token_fee_raw: int = 0
    source: str = "unspecified"
    block_number: int | None = None
    network_fee_native_raw: int | None = None
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_asset("sell_asset", self.sell_asset)
        _require_asset("buy_asset", self.buy_asset)
        _require_raw("sell_amount_raw", self.sell_amount_raw, positive=True)
        _require_raw("expected_buy_amount_raw", self.expected_buy_amount_raw, positive=True)
        _require_raw("buy_token_fee_raw", self.buy_token_fee_raw)
        minimum = self.minimum_buy_amount_raw
        if minimum is None:
            minimum = self.expected_buy_amount_raw
            object.__setattr__(self, "minimum_buy_amount_raw", minimum)
        _require_raw("minimum_buy_amount_raw", minimum, positive=True)
        if minimum > self.expected_buy_amount_raw:
            raise QuoteValidationError(
                "minimum_buy_amount_raw cannot exceed expected_buy_amount_raw"
            )
        if self.block_number is not None:
            _require_raw("block_number", self.block_number)
        if self.network_fee_native_raw is not None:
            _require_raw("network_fee_native_raw", self.network_fee_native_raw)
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, str) for issue in self.issues
        ):
            raise TypeError("issues must be a tuple of strings")

    @property
    def slippage_buffer_raw(self) -> int:
        assert self.minimum_buy_amount_raw is not None
        return self.expected_buy_amount_raw - self.minimum_buy_amount_raw

    @property
    def gross_buy_amount_before_quote_fees_raw(self) -> int:
        return self.expected_buy_amount_raw + self.buy_token_fee_raw


@runtime_checkable
class QuoteProvider(Protocol):
    """Interface for a read-only collateral-to-debt quote source."""

    def get_quote(self, request: QuoteRequest) -> SwapQuote:
        """Return a quote or raise :class:`QuoteUnavailableError`."""


@dataclass(frozen=True, slots=True)
class FixedQuoteProvider:
    """Deterministic provider for fixed snapshots, tests, and reproducibility."""

    fixed_quote: SwapQuote

    def get_quote(self, request: QuoteRequest) -> SwapQuote:
        _validate_quote_matches_request(request, self.fixed_quote)
        return self.fixed_quote


@dataclass(slots=True)
class ZeroXQuoteProvider:
    """Read-only 0x Swap API v2 AllowanceHolder firm-pricing adapter.

    The adapter requests firm pricing but intentionally discards executable
    transaction calldata. It never sets allowances or sends a transaction.
    ``http_client`` can be injected for tests; otherwise ``httpx`` is imported
    lazily. API failures are surfaced as ``QuoteUnavailableError``.
    """

    api_key: str | None = None
    default_taker: str | None = None
    slippage_bps: int = 100
    timeout_seconds: float = 10.0
    base_url: str = "https://api.0x.org"
    http_client: Any | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ZEROX_API_KEY")
        if self.default_taker is not None:
            _require_asset("default_taker", self.default_taker)
        _require_raw("slippage_bps", self.slippage_bps)
        if self.slippage_bps > PERCENTAGE_FACTOR:
            raise PnLError("slippage_bps must be <= 10000")
        if self.timeout_seconds <= 0:
            raise PnLError("timeout_seconds must be positive")
        if not self.base_url.startswith("https://"):
            raise PnLError("base_url must use https")

    def get_quote(self, request: QuoteRequest) -> SwapQuote:
        api_key = self.api_key or os.getenv("ZEROX_API_KEY")
        if not api_key:
            raise QuoteUnavailableError("0x quote unavailable: ZEROX_API_KEY is not configured")
        taker = request.taker or self.default_taker
        if not taker:
            raise QuoteUnavailableError(
                "0x firm pricing quote unavailable: a taker address is required"
            )

        url = f"{self.base_url.rstrip('/')}/swap/allowance-holder/quote"
        params = {
            "chainId": str(request.chain_id),
            "sellToken": request.sell_asset,
            "buyToken": request.buy_asset,
            "sellAmount": str(request.sell_amount_raw),
            "taker": taker,
            "slippageBps": str(self.slippage_bps),
        }
        headers = {
            "0x-api-key": api_key,
            "0x-version": "v2",
            "Accept": "application/json",
        }

        if self.http_client is not None:
            response = self._request(self.http_client, url, params, headers)
        else:
            try:
                import httpx
            except ImportError as exc:
                raise QuoteUnavailableError(
                    "0x quote unavailable: install the optional 'httpx' dependency"
                ) from exc
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = self._request(client, url, params, headers)
            except QuoteError:
                raise
            except Exception as exc:  # pragma: no cover - depends on network stack
                raise QuoteUnavailableError(
                    f"0x quote request failed: {type(exc).__name__}: {exc}"
                ) from exc

        payload = _response_json(response)
        return _parse_zerox_quote(payload, request)

    def _request(
        self,
        client: Any,
        url: str,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> Any:
        try:
            response = client.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response
        except QuoteError:
            raise
        except Exception as exc:
            status = getattr(locals().get("response", None), "status_code", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            raise QuoteUnavailableError(f"0x quote request failed{suffix}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LiquidatorPnLInputs:
    """Inputs whose costs are all normalized to the debt token."""

    collateral_asset: str
    debt_asset: str
    collateral_received_raw: int
    debt_principal_raw: int
    collateral_decimals: int
    debt_decimals: int
    flash_loan_premium_bps: int
    gas_cost_debt_raw: int
    other_cost_debt_raw: int = 0
    chain_id: int = 1
    taker: str | None = None
    use_minimum_swap_proceeds: bool = True

    def __post_init__(self) -> None:
        _require_asset("collateral_asset", self.collateral_asset)
        _require_asset("debt_asset", self.debt_asset)
        _require_raw("collateral_received_raw", self.collateral_received_raw, positive=True)
        _require_raw("debt_principal_raw", self.debt_principal_raw, positive=True)
        token_unit(self.collateral_decimals)
        token_unit(self.debt_decimals)
        _require_raw("flash_loan_premium_bps", self.flash_loan_premium_bps)
        if self.flash_loan_premium_bps > PERCENTAGE_FACTOR:
            raise PnLError("flash_loan_premium_bps must be <= 10000")
        _require_raw("gas_cost_debt_raw", self.gas_cost_debt_raw)
        _require_raw("other_cost_debt_raw", self.other_cost_debt_raw)
        _require_raw("chain_id", self.chain_id, positive=True)
        if self.taker is not None:
            _require_asset("taker", self.taker)
        if not isinstance(self.use_minimum_swap_proceeds, bool):
            raise TypeError("use_minimum_swap_proceeds must be bool")

    @classmethod
    def from_liquidation(
        cls,
        liquidation: LiquidationResult,
        *,
        collateral_asset: str,
        debt_asset: str,
        collateral_decimals: int,
        debt_decimals: int,
        flash_loan_premium_bps: int,
        gas_cost_debt_raw: int,
        other_cost_debt_raw: int = 0,
        chain_id: int = 1,
        taker: str | None = None,
        use_minimum_swap_proceeds: bool = True,
    ) -> LiquidatorPnLInputs:
        return cls(
            collateral_asset=collateral_asset,
            debt_asset=debt_asset,
            collateral_received_raw=liquidation.collateral_to_liquidator_raw,
            debt_principal_raw=liquidation.actual_debt_to_liquidate_raw,
            collateral_decimals=collateral_decimals,
            debt_decimals=debt_decimals,
            flash_loan_premium_bps=flash_loan_premium_bps,
            gas_cost_debt_raw=gas_cost_debt_raw,
            other_cost_debt_raw=other_cost_debt_raw,
            chain_id=chain_id,
            taker=taker,
            use_minimum_swap_proceeds=use_minimum_swap_proceeds,
        )


@dataclass(frozen=True, slots=True)
class LiquidatorPnLResult:
    """Cost decomposition in raw debt token units.

    ``quote_network_fee_native_raw`` preserves the quote provider's native-token
    network-fee estimate as metadata. It is deliberately not added to the P&L:
    callers must supply a full-transaction gas budget already converted to debt
    units through ``LiquidatorPnLInputs.gas_cost_debt_raw``.
    """

    debt_decimals: int
    collateral_decimals: int
    collateral_sold_raw: int
    gross_swap_proceeds_raw: int
    expected_swap_proceeds_raw: int
    swap_proceeds_raw: int
    flash_principal_raw: int
    flash_premium_raw: int
    gas_cost_raw: int
    other_cost_raw: int
    slippage_cost_raw: int
    quote_fee_raw: int
    total_costs_including_principal_raw: int
    net_pnl_raw: int
    break_even_net_swap_proceeds_raw: int
    break_even_gross_swap_proceeds_raw: int
    break_even_debt_per_collateral: Decimal
    quote_source: str
    quote_block_number: int | None
    quote_network_fee_native_raw: int | None
    quote_issues: tuple[str, ...]

    @property
    def profitable(self) -> bool:
        return self.net_pnl_raw > 0

    @property
    def net_pnl(self) -> Decimal:
        sign = -1 if self.net_pnl_raw < 0 else 1
        return Decimal(sign) * raw_to_decimal(abs(self.net_pnl_raw), self.debt_decimals)


def build_quote_request(inputs: LiquidatorPnLInputs) -> QuoteRequest:
    """Build the normalized quote request (pure and deterministic)."""

    return QuoteRequest(
        sell_asset=inputs.collateral_asset,
        buy_asset=inputs.debt_asset,
        sell_amount_raw=inputs.collateral_received_raw,
        sell_decimals=inputs.collateral_decimals,
        buy_decimals=inputs.debt_decimals,
        chain_id=inputs.chain_id,
        taker=inputs.taker,
    )


def calculate_liquidator_pnl(inputs: LiquidatorPnLInputs, quote: SwapQuote) -> LiquidatorPnLResult:
    """Purely calculate net P&L from a previously acquired quote."""

    request = build_quote_request(inputs)
    _validate_quote_matches_request(request, quote)
    assert quote.minimum_buy_amount_raw is not None

    swap_proceeds_raw = (
        quote.minimum_buy_amount_raw
        if inputs.use_minimum_swap_proceeds
        else quote.expected_buy_amount_raw
    )
    slippage_cost_raw = quote.expected_buy_amount_raw - swap_proceeds_raw
    flash_premium_raw = percent_mul_ceil(inputs.debt_principal_raw, inputs.flash_loan_premium_bps)
    gross_swap_proceeds_raw = quote.gross_buy_amount_before_quote_fees_raw

    total_costs_including_principal_raw = (
        inputs.debt_principal_raw
        + flash_premium_raw
        + inputs.gas_cost_debt_raw
        + inputs.other_cost_debt_raw
        + slippage_cost_raw
        + quote.buy_token_fee_raw
    )
    net_pnl_raw = gross_swap_proceeds_raw - total_costs_including_principal_raw

    break_even_net_swap_proceeds_raw = (
        inputs.debt_principal_raw
        + flash_premium_raw
        + inputs.gas_cost_debt_raw
        + inputs.other_cost_debt_raw
    )
    break_even_gross_swap_proceeds_raw = (
        break_even_net_swap_proceeds_raw + slippage_cost_raw + quote.buy_token_fee_raw
    )
    collateral_sold = raw_to_decimal(inputs.collateral_received_raw, inputs.collateral_decimals)
    break_even_gross = raw_to_decimal(break_even_gross_swap_proceeds_raw, inputs.debt_decimals)

    return LiquidatorPnLResult(
        debt_decimals=inputs.debt_decimals,
        collateral_decimals=inputs.collateral_decimals,
        collateral_sold_raw=inputs.collateral_received_raw,
        gross_swap_proceeds_raw=gross_swap_proceeds_raw,
        expected_swap_proceeds_raw=quote.expected_buy_amount_raw,
        swap_proceeds_raw=swap_proceeds_raw,
        flash_principal_raw=inputs.debt_principal_raw,
        flash_premium_raw=flash_premium_raw,
        gas_cost_raw=inputs.gas_cost_debt_raw,
        other_cost_raw=inputs.other_cost_debt_raw,
        slippage_cost_raw=slippage_cost_raw,
        quote_fee_raw=quote.buy_token_fee_raw,
        total_costs_including_principal_raw=total_costs_including_principal_raw,
        net_pnl_raw=net_pnl_raw,
        break_even_net_swap_proceeds_raw=break_even_net_swap_proceeds_raw,
        break_even_gross_swap_proceeds_raw=break_even_gross_swap_proceeds_raw,
        break_even_debt_per_collateral=break_even_gross / collateral_sold,
        quote_source=quote.source,
        quote_block_number=quote.block_number,
        quote_network_fee_native_raw=quote.network_fee_native_raw,
        quote_issues=quote.issues,
    )


def estimate_liquidator_pnl(
    inputs: LiquidatorPnLInputs, quote_provider: QuoteProvider
) -> LiquidatorPnLResult:
    """Acquire a read-only quote, then delegate to the pure P&L function."""

    quote = quote_provider.get_quote(build_quote_request(inputs))
    return calculate_liquidator_pnl(inputs, quote)


def _validate_quote_matches_request(request: QuoteRequest, quote: SwapQuote) -> None:
    if quote.sell_asset.lower() != request.sell_asset.lower():
        raise QuoteValidationError("quote sell asset does not match request")
    if quote.buy_asset.lower() != request.buy_asset.lower():
        raise QuoteValidationError("quote buy asset does not match request")
    if quote.sell_amount_raw != request.sell_amount_raw:
        raise QuoteValidationError("quote sell amount does not match request")


def _response_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception as exc:
        raise QuoteUnavailableError("0x quote returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise QuoteUnavailableError("0x quote returned a non-object JSON response")
    return payload


def _parse_int_field(payload: dict[str, Any], name: str) -> int:
    try:
        value = int(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise QuoteUnavailableError(f"0x quote missing/invalid '{name}'") from exc
    if value < 0:
        raise QuoteUnavailableError(f"0x quote field '{name}' must be non-negative")
    return value


def _parse_zerox_quote(payload: dict[str, Any], request: QuoteRequest) -> SwapQuote:
    if payload.get("liquidityAvailable") is not True:
        raise QuoteUnavailableError("0x quote unavailable: no route liquidity")

    sell_asset = str(payload.get("sellToken", ""))
    buy_asset = str(payload.get("buyToken", ""))
    sell_amount_raw = _parse_int_field(payload, "sellAmount")
    expected_buy_amount_raw = _parse_int_field(payload, "buyAmount")
    minimum_buy_amount_raw = _parse_int_field(payload, "minBuyAmount")

    preliminary = SwapQuote(
        sell_asset=sell_asset,
        buy_asset=buy_asset,
        sell_amount_raw=sell_amount_raw,
        expected_buy_amount_raw=expected_buy_amount_raw,
        minimum_buy_amount_raw=minimum_buy_amount_raw,
    )
    _validate_quote_matches_request(request, preliminary)

    buy_token_fee_raw = 0
    fees = payload.get("fees", {})
    if fees is None:
        fees = {}
    if not isinstance(fees, dict):
        raise QuoteUnavailableError("0x quote returned malformed 'fees'")
    for fee_name, fee in fees.items():
        if fee is None:
            continue
        if not isinstance(fee, dict):
            raise QuoteUnavailableError(f"0x quote returned malformed fee '{fee_name}'")
        try:
            amount = int(fee.get("amount", 0))
        except (TypeError, ValueError) as exc:
            raise QuoteUnavailableError(
                f"0x quote returned invalid fee amount for '{fee_name}'"
            ) from exc
        if amount < 0:
            raise QuoteUnavailableError(f"0x quote returned negative fee amount for '{fee_name}'")
        if amount == 0:
            continue
        fee_token = str(fee.get("token", ""))
        if fee_token.lower() != request.buy_asset.lower():
            raise QuoteUnavailableError(
                "0x quote has a non-buy-token fee that cannot be safely "
                f"normalized: {fee_name} in {fee_token or 'unknown token'}"
            )
        buy_token_fee_raw += amount

    issues_object = payload.get("issues", {})
    if issues_object is None:
        issues_object = {}
    if not isinstance(issues_object, dict):
        raise QuoteUnavailableError("0x quote returned malformed 'issues'")
    issues = tuple(
        sorted(name for name, value in issues_object.items() if value not in (None, False, [], {}))
    )

    block_number = None
    if payload.get("blockNumber") is not None:
        block_number = _parse_int_field(payload, "blockNumber")
    network_fee_native_raw = None
    if payload.get("totalNetworkFee") is not None:
        network_fee_native_raw = _parse_int_field(payload, "totalNetworkFee")

    return SwapQuote(
        sell_asset=sell_asset,
        buy_asset=buy_asset,
        sell_amount_raw=sell_amount_raw,
        expected_buy_amount_raw=expected_buy_amount_raw,
        minimum_buy_amount_raw=minimum_buy_amount_raw,
        buy_token_fee_raw=buy_token_fee_raw,
        source="0x-v2-read-only-firm-pricing",
        block_number=block_number,
        network_fee_native_raw=network_fee_native_raw,
        issues=issues,
    )


__all__ = [
    "FixedQuoteProvider",
    "LiquidatorPnLInputs",
    "LiquidatorPnLResult",
    "PnLError",
    "QuoteError",
    "QuoteProvider",
    "QuoteRequest",
    "QuoteUnavailableError",
    "QuoteValidationError",
    "SwapQuote",
    "ZeroXQuoteProvider",
    "build_quote_request",
    "calculate_liquidator_pnl",
    "estimate_liquidator_pnl",
    "raw_to_decimal",
]
