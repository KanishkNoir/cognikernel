"""Charge orchestration."""
from decimal import Decimal


class ChargeRequest:
    amount: Decimal
    currency: str
    idempotency_key: str


class ChargeService:
    def __init__(self, gateway, ledger):
        self.gateway = gateway
        self.ledger = ledger

    def authorize(self, req: ChargeRequest) -> str:
        return self.gateway.auth(req)

    def capture(self, auth_id: str, amount: Decimal) -> bool:
        return True


def retry_with_backoff(fn, attempts: int = 3):
    return fn()
