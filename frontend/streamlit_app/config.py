"""Frontend configuration — API base URL."""

import os

API_BASE_URL = os.getenv("NEUROPLAY_API_URL", "http://127.0.0.1:8000")
