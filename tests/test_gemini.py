
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True
)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


if not API_KEY:
    raise RuntimeError("Gemini API key not found.")


client = genai.Client(api_key=API_KEY)

print("API key loaded:", bool(API_KEY))
print("Model selected:", MODEL)


# Retry configuration
MAX_ATTEMPTS = 3


for attempt in range(1, MAX_ATTEMPTS + 1):

    try:

        print(f"\nAttempt {attempt}...")

        response = client.models.generate_content(
            model=MODEL,
            contents=(
                "Reply with exactly: "
                "Gemini API connection successful!"
            )
        )

        print("\nGemini response:")
        print(response.text)

        print("\nAPI connection successful!")
        break

    except Exception as error:

        error_message = str(error)

        # Retry only temporary server errors.
        if "503" in error_message and attempt < MAX_ATTEMPTS:

            wait_time = 2 ** attempt

            print(
                f"Gemini is temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

        else:

            print("\nGemini connection failed.")
            print(type(error).__name__)
            print(error)

            break
