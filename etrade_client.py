"""E*TRADE OAuth + trading client (Equities only).

This module is intentionally isolated from the signal engines so we can
enable/disable auto-exec without touching existing alert logic.

We use the official E*TRADE REST API (OAuth 1.0a). The flow is:
  1) get request token
  2) user authorizes -> verifier
  3) exchange for access token

Order workflow per E*TRADE docs:
  - Preview Order: POST /v1/accounts/{accountIdKey}/orders/preview
  - Place Order:   POST /v1/accounts/{accountIdKey}/orders/place
  - List Orders:   GET  /v1/accounts/{accountIdKey}/orders
  - Cancel Order:  PUT  /v1/accounts/{accountIdKey}/orders/cancel

Docs (sandbox base): https://apisb.etrade.com/docs/api/order/api-order-v1.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import datetime
import requests
from zoneinfo import ZoneInfo
from urllib.parse import parse_qs


try:
    # Prefer requests-oauthlib when available.
    from requests_oauthlib import OAuth1Session  # type: ignore

    _HAS_OAUTH = True
except Exception:
    OAuth1Session = None  # type: ignore
    _HAS_OAUTH = False


SANDBOX_BASE = "https://apisb.etrade.com"
LIVE_BASE = "https://api.etrade.com"

# OAuth endpoints have historically been hosted on different base domains depending on
# environment and E*TRADE platform routing. Some environments return HTML 404s if the
# wrong base is used. We therefore try a small, ordered set of known bases.
OAUTH_BASES_SANDBOX = [
    "https://apisb.etrade.com",   # documented sandbox host for REST + often OAuth
    "https://api.etrade.com",     # documented live OAuth host (sometimes handles sandbox keys too)
    "https://etws.etrade.com",    # legacy/alternate OAuth host used by many libraries
    "https://etwssandbox.etrade.com",  # legacy sandbox host (rare)
]
OAUTH_BASES_LIVE = [
    "https://api.etrade.com",     # documented live host
    "https://etws.etrade.com",    # legacy/alternate OAuth host
]



@dataclass
class OAuthTokens:
    oauth_token: str
    oauth_token_secret: str


class ETradeClient:
    """Thin E*TRADE REST client.

    Notes:
      - Equities only for now.
      - Bracket/OCO is NOT supported by E*TRADE API, so stop/TP are managed by our app.
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        sandbox: bool = True,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
    ) -> None:
        if not _HAS_OAUTH:
            raise RuntimeError(
                "requests-oauthlib is required for E*TRADE integration. "
                "Add 'requests-oauthlib' to requirements.txt"
            )

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox = bool(sandbox)
        self.base = SANDBOX_BASE if self.sandbox else LIVE_BASE

        self._session = OAuth1Session(
            client_key=consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret,
            signature_method="HMAC-SHA1",
        )

    # -------------------------
    # OAuth flow
    # -------------------------
    # --- Order status normalization (E*TRADE canonical language) ---
    _STATUS_PARAM_ALIASES = {
        # Common spelling variants / internal shorthands -> E*TRADE canonical
        'CANCELED': 'CANCELLED',
        'CANCELLED': 'CANCELLED',
        'CANCEL': 'CANCELLED',
        'CANCELLED_REQUESTED': 'CANCEL_REQUESTED',
        'CANCEL_REQUEST': 'CANCEL_REQUESTED',
        'CANCELREQUESTED': 'CANCEL_REQUESTED',
        'CANCEL_PENDING': 'CANCEL_REQUESTED',
        'CANCEL_REQUESTED': 'CANCEL_REQUESTED',
        'PARTIAL': 'INDIVIDUAL_FILLS',
        'PARTIALLY_FILLED': 'INDIVIDUAL_FILLS',
        'PARTIALLY_EXECUTED': 'INDIVIDUAL_FILLS',
        'INDIVIDUAL_FILLS': 'INDIVIDUAL_FILLS',
        'FILLED': 'EXECUTED',
        'EXECUTED': 'EXECUTED',
        'OPEN': 'OPEN',
        'EXPIRED': 'EXPIRED',
        'REJECTED': 'REJECTED',
        'UNKNOWN': 'UNKNOWN',
    }

    def _normalize_order_status_param(self, status: str) -> str:
        """Normalize a List Orders `status` query param to E*TRADE canonical values."""
        s = str(status or 'OPEN').strip().upper()
        return self._STATUS_PARAM_ALIASES.get(s, s)

    def _normalize_broker_status(self, status: str) -> str:
        """Normalize broker-returned status strings to E*TRADE canonical values."""
        s = str(status or 'UNKNOWN').strip().upper()
        return self._STATUS_PARAM_ALIASES.get(s, s)

    def _json_or_empty(self, r, *, empty=None, context: str = ""):
        """Return parsed JSON or an empty placeholder for 204/empty bodies.

        E*TRADE sometimes returns 204 No Content (e.g., no records). Treat that as
        an empty response instead of raising JSONDecodeError. If the response is
        non-JSON, raise a readable error that includes a short body snippet.
        """
        if empty is None:
            empty = {}
        try:
            status = getattr(r, "status_code", None)
            text = getattr(r, "text", "") or ""
            ct = (getattr(r, "headers", {}) or {}).get("content-type", "") or ""
            ct_l = ct.lower()

            # 204 No Content is a documented success case for some endpoints.
            if status == 204:
                return empty
            if not text.strip():
                # Some proxies/services return 200 with empty body; treat as empty.
                return empty

            if "json" not in ct_l:
                raise RuntimeError(
                    f"{context or 'E*TRADE'}: non-JSON response (HTTP {status}, content-type={ct!r}) "
                    f"body_snip={text[:200]!r}"
                )

            try:
                return r.json()
            except Exception as e:
                raise RuntimeError(
                    f"{context or 'E*TRADE'}: JSON parse failed (HTTP {status}, content-type={ct!r}) "
                    f"body_snip={text[:200]!r}"
                ) from e
        except Exception:
            # If something unexpected happens, fail loudly with context.
            raise


    def _today_et(self) -> datetime.date:
        """Return today's date in America/New_York (ET)."""
        try:
            return datetime.datetime.now(ZoneInfo("America/New_York")).date()
        except Exception:
            return datetime.datetime.now().date()

    def _mmddyyyy(self, d: datetime.date) -> str:
        """Format date as MMDDYYYY for E*TRADE query parameters."""
        return d.strftime("%m%d%Y")

    def get_request_token(self, callback_url: str = "oob") -> OAuthTokens:
        """Start OAuth by requesting a temporary request token.

        E*TRADE requires `oauth_callback` to be present in the signed OAuth
        parameter set for the request-token step. Some servers will reject
        requests where the callback is only passed as a plain query parameter.

        Using a dedicated OAuth1Session with `callback_uri` ensures the
        callback is embedded in the Authorization header (and therefore signed).
        """
        bases = OAUTH_BASES_SANDBOX if self.sandbox else OAUTH_BASES_LIVE

        # Use a fresh session for the request-token step so callback_uri is
        # definitely included in the OAuth header.
        tmp = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            callback_uri=callback_url,
            signature_method="HMAC-SHA1",
        )
        # E*TRADE's Authorization module specifies GET for request_token.
        # Some hosts return HTML 404s depending on environment routing, so try a
        # small ordered set of known OAuth bases.
        last_err: Optional[str] = None
        for base in bases:
            url = f"{base}/oauth/request_token"
            try:
                r = tmp.get(url, timeout=20)
                if not r.ok:
                    last_err = f"{base} -> {r.status_code}: {r.text!r}"
                    continue
                qs = parse_qs(r.text)
                ot = (qs.get("oauth_token") or [None])[0]
                ots = (qs.get("oauth_token_secret") or [None])[0]
                if not ot or not ots:
                    last_err = f"{base} -> missing oauth_token fields: {r.text!r}"
                    continue
                return OAuthTokens(str(ot), str(ots))
            except Exception as e:
                last_err = f"{base} -> {type(e).__name__}: {e}"
                continue

        raise RuntimeError(f"Token request failed. Tried bases={bases}. Last error: {last_err}")

    def get_authorize_url(self, request_token: str) -> str:
        return f"https://us.etrade.com/e/t/etws/authorize?key={self.consumer_key}&token={requests.utils.quote(request_token, safe='')}"

    def get_access_token(self, request_token: str, request_token_secret: str, verifier: str) -> OAuthTokens:
        # Need a temporary session bound to the request token.
        tmp = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=request_token,
            resource_owner_secret=request_token_secret,
            verifier=verifier,
            signature_method="HMAC-SHA1",
        )
        bases = OAUTH_BASES_SANDBOX if self.sandbox else OAUTH_BASES_LIVE
        # E*TRADE's Authorization module specifies GET for access_token.
        last_err: Optional[str] = None
        for base in bases:
            url = f"{base}/oauth/access_token"
            try:
                r = tmp.get(url, timeout=20)
                if not r.ok:
                    last_err = f"{base} -> {r.status_code}: {r.text!r}"
                    continue
                qs = parse_qs(r.text)
                ot = (qs.get("oauth_token") or [None])[0]
                ots = (qs.get("oauth_token_secret") or [None])[0]
                if not ot or not ots:
                    last_err = f"{base} -> missing oauth_token fields: {r.text!r}"
                    continue
                return OAuthTokens(str(ot), str(ots))
            except Exception as e:
                last_err = f"{base} -> {type(e).__name__}: {e}"
                continue

        raise RuntimeError(f"Access token request failed. Tried bases={bases}. Last error: {last_err}")

    # -------------------------
    # Accounts / portfolio
    # -------------------------
    def list_accounts(self) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/list.json"
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        return self._json_or_empty(r, empty={}, context="list_accounts")

    def get_portfolio(self, account_id_key: str) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/portfolio.json"
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        return self._json_or_empty(r, empty={}, context="get_portfolio")

    # -------------------------
    # Orders
    # -------------------------
    def list_orders(
        self,
        account_id_key: str,
        status: str = "OPEN",
        count: int = 50,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        marker: Optional[str] = None,
        timeout: int = 20,
    ) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders.json"
        params: Dict[str, Any] = {"status": self._normalize_order_status_param(status), "count": count}
        if symbol:
            params["symbol"] = str(symbol).upper().strip()
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date
        if marker:
            params["marker"] = marker
        r = self._session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return self._json_or_empty(r, empty={}, context="list_orders")

    def cancel_order(self, account_id_key: str, order_id: int) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/cancel.json"
        # Per E*TRADE Order API docs, cancel is a PUT with CancelOrderRequest.
        # Keep payload strictly JSON-serializable.
        payload = {"CancelOrderRequest": {"orderId": int(order_id)}}
        r = self._session.put(url, json=payload, timeout=20)
        r.raise_for_status()
        return self._json_or_empty(r, empty={}, context="cancel_order")


    def get_order_details(self, account_id_key: str, order_id: int, timeout: int = 30) -> Dict[str, Any]:
        """Fetch order details for a specific orderId (best-effort).

        Uses the documented order-details endpoint when available.
        Falls back to scanning list_orders across common statuses if the endpoint
        is not supported in a given environment.
        """
        # Primary: order-details endpoint (best-effort; varies by environment)
        try:
            url = f"{self.base}/v1/accounts/{account_id_key}/orders/{int(order_id)}.json"
            r = self._session.get(url, timeout=timeout)
            if r.status_code == 200:
                return self._json_or_empty(r, empty={}, context="get_order_details")
        except Exception:
            pass

        # Fallback: scan list_orders across likely statuses
        statuses = ["OPEN", "EXECUTED", "INDIVIDUAL_FILLS", "CANCELLED", "CANCEL_REQUESTED", "EXPIRED", "REJECTED"]
        for st in statuses:
            try:
                data = self.list_orders(account_id_key, status=st, count=100, timeout=20)
            except Exception:
                continue
            orders = (
                data.get("OrdersResponse", {})
                    .get("Orders", {})
                    .get("Order", [])
                if isinstance(data, dict) else []
            )
            if isinstance(orders, dict):
                orders = [orders]
            for o in orders:
                try:
                    oid = int(o.get("orderId", o.get("orderID", 0)))
                except Exception:
                    oid = 0
                if oid == int(order_id):
                    return o if isinstance(o, dict) else {"Order": o}
        return {}

    def get_order_filled_and_avg_price(self, account_id_key: str, order_id: int) -> tuple[int, Optional[float]]:
        """Return (filled_qty, avg_execution_price) best-effort.

        This is used for realized P&L reporting; execution safety does NOT depend on it.
        """
        data: Dict[str, Any] = {}
        try:
            data = self.get_order_details(account_id_key, int(order_id))
        except Exception:
            data = {}

        filled = 0
        avg_price: Optional[float] = None

        def _walk(obj):
            nonlocal filled, avg_price
            if isinstance(obj, dict):
                for k, v in obj.items():
                    kl = str(k).lower()
                    if kl in {"filledquantity", "filledqty", "quantityexecuted", "executedquantity"}:
                        try:
                            filled += int(float(v))
                        except Exception:
                            pass
                    if avg_price is None and kl in {"averageexecutionprice", "avgexecutionprice", "averageprice", "executionprice", "fillprice", "avgprice"}:
                        try:
                            avg_price = float(v)
                        except Exception:
                            pass
                    _walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    _walk(it)

        _walk(data)

        # If avg_price wasn't found, try to infer from limit/stop price fields (last-resort)
        if avg_price is None:
            def _find_first_price(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        kl = str(k).lower()
                        if kl in {"limitprice", "stopprice", "price"}:
                            try:
                                return float(v)
                            except Exception:
                                pass
                        r = _find_first_price(v)
                        if r is not None:
                            return r
                elif isinstance(obj, list):
                    for it in obj:
                        r = _find_first_price(it)
                        if r is not None:
                            return r
                return None
            avg_price = _find_first_price(data)

        try:
            filled = int(filled)
        except Exception:
            filled = 0
        return (filled, avg_price)

    def _extract_preview_id(self, resp: Dict[str, Any]) -> Optional[int]:
        """Best-effort extraction of previewId from PreviewOrder responses."""
        if not isinstance(resp, dict):
            return None

        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "previewid":
                        try:
                            return int(v)
                        except Exception:
                            pass
                    r = _walk(v)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for it in obj:
                    r = _walk(it)
                    if r is not None:
                        return r
            return None

        return _walk(resp)

    def preview_order(self, account_id_key: str, order: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/preview.json"
        # E*TRADE schema expects clientOrderId at the PreviewOrderRequest level (NOT inside Order).
        # Older call sites may have placed it inside the Order dict; normalize here.
        # IMPORTANT: do NOT mutate the caller's `order` dict; we may reuse it for PlaceOrder.
        order_clean: Dict[str, Any] = dict(order) if isinstance(order, dict) else {}
        client_order_id = order_clean.pop("clientOrderId", None)

        por: Dict[str, Any] = {"orderType": "EQ", "Order": [order_clean]}
        if client_order_id:
            por["clientOrderId"] = str(client_order_id)[:20]

        payload = {"PreviewOrderRequest": por}
        r = self._session.post(url, json=payload, timeout=20)
        try:
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"PreviewOrder failed: {e}; body={getattr(r, 'text', '')[:1200]}")
        resp = self._json_or_empty(r, empty={}, context="preview_order")
        if isinstance(resp, dict) and 'Error' in resp:
            err = resp.get('Error') or {}
            # Surface broker error details for diagnostics (do not include secrets).
            code = err.get('code') if isinstance(err, dict) else None
            msg = err.get('message') if isinstance(err, dict) else None
            raise RuntimeError(f"PreviewOrder returned Error: code={code} message={msg} err={err}")
        return resp

    def place_order(self, account_id_key: str, order: Dict[str, Any], preview_id: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/place.json"
        # E*TRADE docs show Order as a LIST and expect PreviewIds when placing.
        if preview_id is None:
            # Autopilot-safety: do NOT attempt to place without an explicit previewId.
            # If preview succeeded but the response shape changed, we'd rather fail loudly
            # than send an ambiguous request that may be rejected or behave unexpectedly.
            raise RuntimeError("Missing previewId for place_order")
        # E*TRADE schema expects clientOrderId at the PlaceOrderRequest level (NOT inside Order).
        # IMPORTANT: do NOT mutate the caller's `order` dict; preview/place may share the same dict.
        order_clean: Dict[str, Any] = dict(order) if isinstance(order, dict) else {}
        client_order_id = order_clean.pop("clientOrderId", None)

        por: Dict[str, Any] = {"orderType": "EQ", "Order": [order_clean]}
        por["PreviewIds"] = [{"previewId": int(preview_id)}]
        if client_order_id:
            por["clientOrderId"] = str(client_order_id)[:20]
        payload = {"PlaceOrderRequest": por}
        r = self._session.post(url, json=payload, timeout=20)
        try:
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"PlaceOrder failed: {e}; body={getattr(r, 'text', '')[:1200]}")
        return self._json_or_empty(r, empty={}, context="place_order")

    # -------------------------
    # Order builders (EQ only)
    # -------------------------
    @staticmethod
    def build_equity_order(
        symbol: str,
        action: str,
        quantity: int,
        price_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        market_session: str = "REGULAR",
        order_term: str = "GOOD_FOR_DAY",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an EQ order dict compatible with preview/place endpoints.

        Notes on time-in-force ("good for day" vs IOC/FOK):
        - For your auto-exec strategy we *explicitly* default to GOOD_FOR_DAY.
        - We do not use IOC/FOK special instructions because your bot manages
          freshness via its own timeout + EOD cleanup.
        """
        # Normalize / guard order term to avoid ambiguous broker defaults.
        # E*TRADE uses orderTerm values like GOOD_FOR_DAY / GOOD_TILL_CANCEL.
        term = str(order_term or "GOOD_FOR_DAY").upper().strip()
        if term in {"DAY", "GFD"}:
            term = "GOOD_FOR_DAY"
        elif term in {"GTC"}:
            term = "GOOD_TILL_CANCEL"
        if term not in {"GOOD_FOR_DAY", "GOOD_TILL_CANCEL"}:
            raise ValueError(f"Unsupported order_term: {order_term}")
        if price_type not in {"MARKET", "LIMIT", "STOP"}:
            raise ValueError(f"Unsupported price_type: {price_type}")
        if price_type == "LIMIT" and limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")
        if price_type == "STOP" and stop_price is None:
            raise ValueError("stop_price required for STOP orders")

        order: Dict[str, Any] = {
            "orderTerm": term,
            "marketSession": market_session,
            "priceType": price_type,
            "allOrNone": False,
            "Instrument": [
                {
                    "Product": {"securityType": "EQ", "symbol": symbol},
                    "orderAction": action,
                    "quantityType": "QUANTITY",
                    "quantity": int(quantity),
                }
            ],
        }
        if client_order_id:
            # E*TRADE supports a developer-supplied clientOrderId (<=20 chars) to help prevent duplicates.
            order["clientOrderId"] = str(client_order_id)[:20]

        # Only include price fields applicable to the order type.
        # Some broker environments are picky about irrelevant fields being present.
        if price_type == "LIMIT":
            order["limitPrice"] = float(limit_price)  # type: ignore[arg-type]
        elif price_type == "STOP":
            order["stopPrice"] = float(stop_price)  # type: ignore[arg-type]
        return order
    # -------------------------
    # Convenience wrappers used by auto_exec.py
    # -------------------------
    def _extract_order_id(self, resp: Dict[str, Any]) -> int:
        """Best-effort extraction of orderId from E*TRADE responses."""
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected response type: {type(resp)}")
        # Common keys
        for key in ("orderId", "OrderId", "orderID", "OrderID"):
            if key in resp and str(resp[key]).isdigit():
                return int(resp[key])
        # PlaceOrderResponse.OrderIds.OrderId[0].orderId
        def _walk(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k.lower() == "orderid" and isinstance(v, (int,str)) and str(v).isdigit():
                        return int(v)
                    r=_walk(v)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for it in obj:
                    r=_walk(it)
                    if r is not None:
                        return r
            return None
        oid=_walk(resp)
        if oid is None:
            raise RuntimeError(f"Could not extract orderId from response keys={list(resp.keys())[:10]}")
        return int(oid)

    def place_equity_limit_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        limit_price: float,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="LIMIT",
            limit_price=float(limit_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        # Preview then place (E*TRADE requires preview before place)
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (LIMIT). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def place_equity_limit_order_ex(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        limit_price: float,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> tuple[int, int]:
        """Like place_equity_limit_order but returns (order_id, preview_id).

        This is used by auto_exec for lifecycle breadcrumbs/auditability.
        """
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="LIMIT",
            limit_price=float(limit_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (LIMIT). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        oid = self._extract_order_id(resp)
        return int(oid), int(pid)

    def place_equity_stop_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        stop_price: float,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="STOP",
            stop_price=float(stop_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (STOP). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def place_equity_stop_order_ex(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        stop_price: float,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> tuple[int, int]:
        """Like place_equity_stop_order but returns (order_id, preview_id)."""
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="STOP",
            stop_price=float(stop_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (STOP). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        oid = self._extract_order_id(resp)
        return int(oid), int(pid)

    def place_equity_market_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="MARKET",
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (MARKET). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def place_equity_market_order_ex(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        action: str,
        client_order_id: Optional[str] = None,
        market_session: str = "REGULAR",
    ) -> tuple[int, int]:
        """Like place_equity_market_order but returns (order_id, preview_id)."""
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="MARKET",
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
            client_order_id=client_order_id,
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        if pid is None:
            raise RuntimeError(f"Preview response missing previewId (MARKET). keys={list(prev.keys())[:12]}")
        resp = self.place_order(account_id_key, order, preview_id=pid)
        oid = self._extract_order_id(resp)
        return int(oid), int(pid)

    def _extract_cancel_event_status(self, obj: Any) -> Optional[str]:
        """Best-effort scan of an order-details/list-orders payload for cancel lifecycle events.

        E*TRADE may expose terminal/transition information in nested event/message fields
        even when orderStatus lags. This helper returns one of:
          CANCELLED, CANCEL_REQUESTED, REJECTED
        or None if no signal is found.
        """
        found = set()

        def _walk(x: Any) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(k, str):
                        ku = k.upper().strip()
                        if ku in {"EVENT", "EVENTNAME", "NAME", "TYPE", "STATUS", "MESSAGE", "MESSAGES", "DESCRIPTION", "TEXT", "CODE"}:
                            if isinstance(v, str):
                                su = v.upper().strip()
                                if "ORDER_CANCELLED" in su or su == "CANCELLED":
                                    found.add("CANCELLED")
                                elif "CANCEL_REJECTED" in su:
                                    found.add("REJECTED")
                                elif "CANCEL_REQUESTED" in su or su == "CANCEL_REQUESTED":
                                    found.add("CANCEL_REQUESTED")
                    _walk(v)
            elif isinstance(x, list):
                for i in x:
                    _walk(i)
            elif isinstance(x, str):
                su = x.upper().strip()
                if "ORDER_CANCELLED" in su or su == "CANCELLED":
                    found.add("CANCELLED")
                elif "CANCEL_REJECTED" in su:
                    found.add("REJECTED")
                elif "CANCEL_REQUESTED" in su or su == "CANCEL_REQUESTED":
                    found.add("CANCEL_REQUESTED")

        _walk(obj)
        if "CANCELLED" in found:
            return "CANCELLED"
        if "REJECTED" in found:
            return "REJECTED"
        if "CANCEL_REQUESTED" in found:
            return "CANCEL_REQUESTED"
        return None

    def get_order_status_and_filled_qty(self, account_id_key: str, order_id: int, symbol: Optional[str] = None) -> tuple[str, int]:
        """Return a simplified status + filled quantity for an orderId.

        Strategy (LIVE-safe, SANDBOX-safe):
          1) Try a deterministic lookup via Order Details endpoint (by orderId).
          2) Fall back to List Orders scanning across likely statuses.

        This improves reliability of status detection (reducing UNKNOWN) without changing
        downstream trading decisions.
        """
        oid_int = int(order_id)

        # --- 1) Deterministic lookup first ---
        try:
            detail = self.get_order_details(account_id_key, oid_int)
        except Exception:
            detail = {}

        found = None

        # Some wrappers/libraries return the order object directly.
        if isinstance(detail, dict):
            try:
                direct_oid = int(detail.get("orderId", detail.get("orderID", 0)) or 0)
            except Exception:
                direct_oid = 0
            if direct_oid == oid_int:
                found = detail

        # OrderDetailsResponse.Order is typically dict or list
        if not found and isinstance(detail, dict):
            orders = (
                detail.get("OrderDetailsResponse", {})
                      .get("Order", [])
                if isinstance(detail.get("OrderDetailsResponse", {}), dict) else []
            )
            if isinstance(orders, dict):
                orders = [orders]
            if isinstance(orders, list):
                for o in orders:
                    try:
                        oid = int(o.get("orderId", o.get("orderID", 0)))
                    except Exception:
                        oid = 0
                    if oid == oid_int:
                        found = o
                        break

        # --- 2) Fallback: List Orders scan (with date range + symbol filter) ---
        if not found:
            statuses = ["OPEN", "EXECUTED", "INDIVIDUAL_FILLS", "CANCELLED", "CANCEL_REQUESTED", "EXPIRED", "REJECTED"]
            today = self._today_et()
            from_dt = self._mmddyyyy(today - datetime.timedelta(days=7))
            to_dt = self._mmddyyyy(today)

            sym = str(symbol or "").strip().upper() or None

            for st in statuses:
                marker = None
                for _page in range(3):  # minimal paging guard
                    try:
                        data = self.list_orders(
                            account_id_key,
                            status=st,
                            count=100,
                            symbol=sym,
                            from_date=from_dt,
                            to_date=to_dt,
                            marker=marker,
                        )
                    except Exception:
                        break

                    orders = (
                        data.get("OrdersResponse", {})
                            .get("Orders", {})
                            .get("Order", [])
                        if isinstance(data, dict) else []
                    )
                    if isinstance(orders, dict):
                        orders = [orders]

                    for o in orders:
                        try:
                            oid = int(o.get("orderId", o.get("orderID", 0)))
                        except Exception:
                            oid = 0
                        if oid == oid_int:
                            found = o
                            break
                    if found:
                        break

                    # Page if marker provided (E*TRADE supports marker pagination)
                    try:
                        marker = (
                            data.get("OrdersResponse", {})
                                .get("marker")
                            if isinstance(data, dict) else None
                        )
                    except Exception:
                        marker = None
                    if not marker:
                        break
                if found:
                    break

        if not found:
            return ("UNKNOWN", 0)

        # Normalize status
        status = self._normalize_broker_status(found.get("orderStatus", found.get("status", "UNKNOWN")))
        event_status = self._extract_cancel_event_status(found)
        if event_status and status in {"UNKNOWN", "OPEN", "CANCEL_REQUESTED"}:
            status = event_status

        # Best-effort filled qty extraction (recursive).
        filled = 0
        _FILLED_KEYS = {
            "filledquantity", "filledqty", "quantityfilled", "filled",
        }
        _EXEC_KEYS = {
            "quantityexecuted", "executedquantity", "totalexecutedquantity",
            "cumulativequantity", "cumqty", "execqty", "executedqty",
        }

        def _sum_qty(obj):
            nonlocal filled
            if isinstance(obj, dict):
                for k, v in obj.items():
                    kl = str(k).lower()
                    if kl in _FILLED_KEYS or kl in _EXEC_KEYS:
                        try:
                            filled += int(float(v))
                        except Exception:
                            pass
                    _sum_qty(v)
            elif isinstance(obj, list):
                for it in obj:
                    _sum_qty(it)

        _sum_qty(found)

        # If status indicates executions but we still didn't extract quantity, infer full fill from ordered qty.
        if filled == 0 and status in ("EXECUTED", "FILLED", "INDIVIDUAL_FILLS"):
            # Try common ordered quantity keys first.
            for k in ("orderedQuantity", "quantity", "qty", "orderQuantity"):
                if k in found:
                    try:
                        filled = int(float(found[k]))
                        break
                    except Exception:
                        pass

            # As a last resort, attempt to extract from nested structures.
            if filled == 0:
                try:
                    def _find_first_order_qty(obj):
                        nonlocal filled
                        if filled:
                            return
                        if isinstance(obj, dict):
                            for kk, vv in obj.items():
                                if str(kk).lower() in {"orderedquantity", "quantity", "qty", "orderquantity"}:
                                    try:
                                        filled = int(float(vv))
                                        return
                                    except Exception:
                                        continue
                                _find_first_order_qty(vv)
                        elif isinstance(obj, list):
                            for it in obj:
                                _find_first_order_qty(it)

                    _find_first_order_qty(found)
                except Exception:
                    pass

        return (status, int(filled))

    def get_positions_map(self, account_id_key: str, market_session: str = "REGULAR") -> Dict[str, int]:
        """Return a simple {symbol: quantity} map for current positions."""
        data = self.get_portfolio(account_id_key)  # already .json
        pos = (
            data.get("PortfolioResponse", {})
                .get("AccountPortfolio", [])
            if isinstance(data, dict) else []
        )
        if isinstance(pos, dict):
            pos = [pos]
        out: Dict[str, int] = {}
        for ap in pos:
            positions = ap.get("Position", [])
            if isinstance(positions, dict):
                positions = [positions]
            for p in positions:
                sym = str(p.get("symbol", p.get("displaySymbol", p.get("Product", {}).get("symbol", "")))).upper()
                if not sym:
                    prod = p.get("Product", {}) if isinstance(p.get("Product", {}), dict) else {}
                    sym = str(prod.get("symbol", "")).upper()
                qty = p.get("quantity", p.get("qty", p.get("positionQuantity", 0)))
                try:
                    out[sym] = int(float(qty))
                except Exception:
                    continue

        # Defensive fallback: API payload shapes vary (especially between sandbox/live).
        # If the standard path yields nothing, walk the whole response and extract
        # anything that looks like a position row.
        if not out and isinstance(data, dict):
            def walk(obj: Any):
                if isinstance(obj, dict):
                    # Heuristic: a "position"-like dict tends to contain symbol + quantity keys.
                    sym = obj.get("symbol") or obj.get("displaySymbol")
                    prod = obj.get("Product") if isinstance(obj.get("Product"), dict) else None
                    if not sym and prod:
                        sym = prod.get("symbol") or prod.get("displaySymbol")
                    qty = obj.get("quantity") or obj.get("qty") or obj.get("positionQuantity")
                    if sym and qty is not None:
                        try:
                            out[str(sym).upper()] = int(float(qty))
                        except Exception:
                            pass
                    for v in obj.values():
                        walk(v)
                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)
            try:
                walk(data)
            except Exception:
                pass
        return out

    def _extract_order_rows(self, data: Any) -> list[Dict[str, Any]]:
        """Normalize list_orders/get_order_details payloads into a list of order dicts."""
        if isinstance(data, list):
            return [o for o in data if isinstance(o, dict)]
        if not isinstance(data, dict):
            return []

        candidates = data.get("OrdersResponse", {}).get("Orders", {}).get("Order", [])
        if isinstance(candidates, dict):
            return [candidates]
        if isinstance(candidates, list):
            return [o for o in candidates if isinstance(o, dict)]

        if data.get("orderId") or data.get("orderID"):
            return [data]
        return []

    def _extract_client_order_id_from_order(self, order: Dict[str, Any]) -> str:
        """Best-effort clientOrderId extraction across E*TRADE response shapes."""
        if not isinstance(order, dict):
            return ""
        direct = order.get("clientOrderId") or order.get("client_order_id")
        if direct not in (None, ""):
            return str(direct).strip()

        for key in ("OrderDetail", "orderDetail"):
            od = order.get(key)
            if isinstance(od, dict):
                od = [od]
            if isinstance(od, list):
                for row in od:
                    if not isinstance(row, dict):
                        continue
                    val = row.get("clientOrderId") or row.get("client_order_id")
                    if val not in (None, ""):
                        return str(val).strip()
        return ""

    def find_order_by_client_order_id(
        self,
        account_id_key: str,
        client_order_id: str,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort lookup of an order using list_orders filtering on clientOrderId.

        E*TRADE sometimes returns "duplicate order" on place, while still accepting the
        original order. When that happens, we may not have the orderId; this helper lets
        us recover it.
        """
        if not client_order_id:
            return None

        cid = str(client_order_id).strip()
        statuses = ["OPEN", "EXECUTED", "INDIVIDUAL_FILLS", "CANCEL_REQUESTED", "CANCELLED", "EXPIRED", "REJECTED"]
        for st in statuses:
            try:
                data = self.list_orders(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    status=st,
                    count=100,
                )
            except Exception:
                continue
            for o in self._extract_order_rows(data):
                try:
                    if self._extract_client_order_id_from_order(o) == cid:
                        return o
                except Exception:
                    continue
        return None