import os
import tempfile

# Must be set before any `app.*` import so the store writes to a throwaway DB.
os.environ["GATEWAY_DB"] = os.path.join(tempfile.mkdtemp(), "test_gateway.db")
os.environ.pop("GROK_API_KEY", None)  # tests must never call the real Grok API
