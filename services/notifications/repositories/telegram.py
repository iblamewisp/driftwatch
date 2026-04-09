import requests

from app.schemas.notifications import DriftAlertPayload
from monitoring.logging import get_logger
from services.notifications.base import AbstractNotificationService

logger = get_logger("notifications.telegram")


class TelegramNotificationService(AbstractNotificationService):

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send_alert(self, payload: DriftAlertPayload) -> None:
        message = (
            f"*Driftwatch Alert*\n\n"
            f"{payload.summary}\n\n"
            f"Baseline: `{payload.baseline_score:.4f}`\n"
            f"Current:  `{payload.current_score:.4f}`\n"
            f"Drop:     `{payload.delta_percent:.1f}%` (threshold: {payload.threshold_percent:.1f}%)\n"
            f"Time:     `{payload.detected_at.isoformat()}`"
        )
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": self._chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("telegram_alert_sent", chat_id=self._chat_id)
