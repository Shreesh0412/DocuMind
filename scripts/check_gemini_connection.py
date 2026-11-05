from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def request_json(url: str, api_key: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"x-goog-api-key": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body[:1000]
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(body)
        except json.JSONDecodeError:
            details = body[:1000]
        return error.code, details
    except URLError as error:
        return None, str(error.reason)


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

    print("DocuMind Gemini access diagnostic")
    print(f"Model configured: {model}")
    print(f"API key loaded: {'yes' if api_key else 'no'} (key is never printed)")
    proxy_configured = any(
        os.getenv(name)
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )
    print(f"Proxy environment variable detected: {'yes' if proxy_configured else 'no'}")

    if not api_key:
        print("\nFAIL: GEMINI_API_KEY is missing. Check the project .env file.")
        return 1

    base_url = "https://generativelanguage.googleapis.com/v1beta"
    print("\n1) Checking API key/project access to the model catalogue...")
    status, catalogue = request_json(f"{base_url}/models?pageSize=100", api_key)
    print(f"Model catalogue HTTP status: {status}")
    if status != 200 or not isinstance(catalogue, dict):
        print(f"Response: {catalogue}")
        print("\nThe model-list request itself is blocked. Check the API key, API restrictions, project status, and network path.")
        return 1

    models = catalogue.get("models", [])
    matching = [
        item for item in models
        if isinstance(item, dict)
        and item.get("name", "").removeprefix("models/") == model
    ]
    if matching:
        methods = matching[0].get("supportedGenerationMethods", [])
        print("Configured model appears in catalogue: yes")
        print(f"Supports generateContent: {'yes' if 'generateContent' in methods else 'not listed'}")
    else:
        print("Configured model appears in first catalogue page: no")
        print("This may mean the model is unavailable to this project or is on a later page.")

    print("\n2) Sending a minimal text-generation request (no JSON schema)...")
    payload = {
        "contents": [{"parts": [{"text": "Reply with exactly: DocuMind test OK"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 32},
    }
    status, result = request_json(
        f"{base_url}/models/{model}:generateContent",
        api_key,
        payload,
    )
    print(f"Generation HTTP status: {status}")

    if status == 200 and isinstance(result, dict):
        try:
            response_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            response_text = "Generation returned HTTP 200, but no text field was found."
        print(f"Generation response: {response_text}")
        print("\nPASS: The API key, project, model, and network can generate text.")
        print("If DocuMind still fails, the remaining issue is likely in the app request/configuration.")
        return 0

    print(f"Response details: {result}")
    detail_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    lowered = detail_text.lower()
    if "<!doctype html" in lowered or "your client does not have permission" in lowered:
        print("\nDIAGNOSIS: Google returned an HTML 403 at its frontend. Retry with VPN/proxy disabled and from a different network (for example, a phone hotspot).")
    elif status == 403:
        print("\nDIAGNOSIS: Google's API denied access. Check project availability, API-key restrictions, and model access in AI Studio.")
    elif status == 400:
        print("\nDIAGNOSIS: Google rejected the test request. Inspect the API error details above.")
    elif status == 429:
        print("\nDIAGNOSIS: The project has hit a rate or quota limit. Check AI Studio's Rate limits page.")
    else:
        print("\nDIAGNOSIS: Check the status and response above, then retry after verifying network connectivity.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
