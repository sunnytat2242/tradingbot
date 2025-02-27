import asyncio

import telegram

from optionsbotEnhancedVersion1 import options_stocks

# Replace with your Telegram Bot Token and Chat ID (using '@' if it's a channel)
TELEGRAM_BOT_TOKEN = "7787739717:AAEVjjvGOTu2p-oy03mnzl9BT84um76BKbQ"
TELEGRAM_CHAT_ID = "-4743117490"  # Note the '@' prefix for channels

async def send_telegram_alert(message):
    """
    Sends an alert to a Telegram chat.
    :param message: The alert message to be sent.
    """
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        response = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print("Telegram alert sent successfully!")
    except Exception as e:
        print("Error sending Telegram alert:", e)

async def get_updates():
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        updates = await bot.get_updates()
        for update in updates:
            print(update)
    except Exception as e:
        print("Error getting updates:", e)
# Example usage:
if __name__ == '__main__':
    asyncio.run(get_updates())
    asyncio.run(send_telegram_alert(f"This is the list its monitoring {options_stocks}"))
