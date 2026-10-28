# Anti-Mock Patterns & Golden Fixture Reference

## The Core Problem with Synthetic Mocks
When engineers write unit tests with mocks like:
```python
mock_resp.json.return_value = {"ports": [{"privatePort": 8000, "type": "http"}]}
```
they are testing **their own mental model** of the third-party API, not the actual third-party API.
When the API returns:
```json
{"ports": ["8000/http"], "runtime": null}
```
the unit tests pass with 100% green checkmarks, but the application crashes in production.

---

## Best Practices

### 1. Golden Fixture Directory
Keep raw snapshots of real API interactions under `tests/fixtures/`:
```text
tests/fixtures/
├── runpod_v2_pods_running.json
├── runpod_v2_pods_stopped.json
├── runpod_v2_pods_string_ports.json
└── runpod_sse_logs_sample.txt
```

### 2. Loading Golden Fixtures in Tests
```python
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def load_fixture(filename: str):
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)
```

### 3. Pipeline Ingestion Testing
Instead of mocking the internal calls between steps, feed the golden fixture directly into the ingestion method and assert that the downstream business state (e.g. `engine.proxy_url` or database record) is correctly established.
