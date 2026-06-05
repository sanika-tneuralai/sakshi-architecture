"""
Telegram alert channel.

The alert service is a generic dispatcher: process_pipeline_alerts() decides
*that* an alert should fire; this module is one of the channels it fans out to.

A single Bot API sendMessage call delivers a fired alert to a chat/group. The
channel is opt-in and self-disabling: if TELEGRAM_ENABLED is false, or the bot
token / chat id are missing, send_telegram() is a guarded no-op so the rest of
the alert pipeline runs unchanged. Any delivery failure is logged and swallowed
— Telegram being down must never break alert processing or the HTTP response.

Required env to actually send:
  TELEGRAM_ENABLED=true
  TELEGRAM_BOT_TOKEN=<token from @BotFather>
  TELEGRAM_CHAT_ID=<numeric chat id or @channelusername>
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() in ("1", "true", "yes")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT", "5.0"))

# Comma-separated allowlist of usecase_ids that should go to Telegram. Keeps
# high-frequency vision alerts (parking, gun, etc.) off the channel while we
# start with camera-health alerts. Empty/"*" => every fired alert is sent.
_raw_allow = os.getenv("TELEGRAM_USECASES", "camera_offline,camera_recovered").strip()
TELEGRAM_USECASES = (
    None if _raw_allow in ("", "*")
    else {u.strip() for u in _raw_allow.split(",") if u.strip()}
)


def is_telegram_target(usecase_id: str) -> bool:
    """True if this usecase should be delivered to Telegram, per the allowlist."""
    if not TELEGRAM_ENABLED:
        return False
    if TELEGRAM_USECASES is None:
        return True
    return usecase_id in TELEGRAM_USECASES


def send_telegram(text: str) -> bool:
    """Send a plain-text message to the configured Telegram chat.

    Returns True on a confirmed send, False otherwise (disabled, unconfigured,
    or a delivery error). Never raises.
    """
    if not TELEGRAM_ENABLED:
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[TELEGRAM] enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=TELEGRAM_TIMEOUT,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("[TELEGRAM] message delivered")
            return True
        logger.warning(f"[TELEGRAM] send failed: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"[TELEGRAM] send error: {e}")
        return False
