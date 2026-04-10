from abc import ABC, abstractmethod

from app.schemas.notifications import DriftAlertPayload


class AbstractNotificationService(ABC):

    @abstractmethod
    async def send_alert(self, payload: DriftAlertPayload) -> None:
        """Send drift alert. Raise on failure."""
        ...


def get_notification_service(channel: str) -> AbstractNotificationService | None:
    if channel == "telegram":
        from services.notifications.repositories.telegram import TelegramNotificationService
        from app.config import settings
        return TelegramNotificationService(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
    if channel == "webhook":
        from services.notifications.repositories.webhook import WebhookNotificationService
        from app.config import settings
        return WebhookNotificationService(
            url=settings.WEBHOOK_URL,
            secret=settings.WEBHOOK_SECRET,
        )
    return None
