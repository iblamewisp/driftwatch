"""
Async circuit breaker.

States:
  CLOSED    — normal, all calls pass through
  OPEN      — fail_max consecutive failures hit, calls rejected immediately
  HALF_OPEN — reset_timeout elapsed, one probe call allowed through;
              success → CLOSED, failure → OPEN again
"""

import time
from enum import Enum

from monitoring.logging import get_logger

logger = get_logger("circuit_breaker")


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""
    def __init__(self, name: str):
        super().__init__(f"Circuit breaker '{name}' is OPEN — service unavailable")
        self.breaker_name = name


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:

    def __init__(
        self,
        name: str,
        fail_max: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout

        self._state = _State.CLOSED
        self._fail_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> _State:
        if self._state == _State.OPEN:
            assert self._opened_at is not None
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                self._state = _State.HALF_OPEN
                logger.info("circuit_breaker_half_open", name=self.name)
        return self._state

    def _on_success(self) -> None:
        if self._state != _State.CLOSED:
            logger.info("circuit_breaker_closed", name=self.name)
        self._fail_count = 0
        self._opened_at = None
        self._state = _State.CLOSED

    def _on_failure(self, exc: BaseException) -> None:
        self._fail_count += 1
        logger.warning(
            "circuit_breaker_failure",
            name=self.name,
            fail_count=self._fail_count,
            fail_max=self.fail_max,
            error=str(exc),
        )
        if self._fail_count >= self.fail_max:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "circuit_breaker_opened",
                name=self.name,
                reset_in_seconds=self.reset_timeout,
            )

    async def call(self, coro):
        """
        Await coro through the circuit breaker.
        Raises CircuitBreakerOpen if the circuit is OPEN.
        Re-raises the original exception on failure (after recording it).
        """
        if self.state == _State.OPEN:
            raise CircuitBreakerOpen(self.name)

        try:
            result = await coro
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise


# ── Singleton breakers — one per external dependency ─────────────────────────

litserve_breaker = CircuitBreaker(
    name="litserve",
    fail_max=5,
    reset_timeout=30.0,
)

# Per-provider breakers — keyed by provider name
_provider_breakers: dict[str, CircuitBreaker] = {
    "openai": CircuitBreaker(name="openai", fail_max=3, reset_timeout=60.0),
    "anthropic": CircuitBreaker(name="anthropic", fail_max=3, reset_timeout=60.0),
}

# Legacy alias kept for streaming path that references openai_breaker directly
openai_breaker = _provider_breakers["openai"]

redis_breaker = CircuitBreaker(
    name="redis",
    fail_max=5,
    reset_timeout=20.0,
)


def get_provider_breaker(provider: str) -> CircuitBreaker:
    """Return the circuit breaker for a given provider name."""
    if provider not in _provider_breakers:
        _provider_breakers[provider] = CircuitBreaker(name=provider, fail_max=3, reset_timeout=60.0)
    return _provider_breakers[provider]
