import hashlib
import hmac

import httpx

from app.schemas.notifications import DriftAlertPayload
from monitoring.logging import get_logger
from services.notifications.base import AbstractNotificationService

logger = get_logger("notifications.webhook")


class WebhookNotificationService(AbstractNotificationService):

    def __init__(self, url: str, secret: str) -> None:
        self._url = url
        self._secret = secret

    async def send_alert(self, payload: DriftAlertPayload) -> None:
        body = payload.model_dump_json()
        signature = hmac.new(
            self._secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self._url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Driftwatch-Signature": signature,
                },
            )
            response.raise_for_status()
        logger.info("webhook_alert_sent", url=self._url)
