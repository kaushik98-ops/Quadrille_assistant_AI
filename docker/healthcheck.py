"""Container health probe: succeeds only if the API answers /health with 200."""
import os
import sys
import urllib.request

port = os.environ.get("API_PORT", "8000")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
