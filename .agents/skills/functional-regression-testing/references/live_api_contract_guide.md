# Live API Contract Smoke Testing Blueprint

## Overview
Live API smoke tests run genuine HTTP/REST/WebSocket calls against real external services when environment credentials are present. They verify that the live cloud API hasn't introduced breaking schema changes, altered rate-limit policies, or changed status payloads.

---

## Pattern: Conditional Opt-In Testing

Tests must NEVER fail or block in offline CI environments or for developers without paid cloud credentials.

### Python (`unittest`)
```python
import os
import unittest
import urllib.request
import json

@unittest.skipUnless(os.getenv("EXTERNAL_SERVICE_API_KEY"), "Skipping live smoke test: EXTERNAL_SERVICE_API_KEY not set")
class TestLiveServiceAPIContract(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("EXTERNAL_SERVICE_API_KEY")

    def test_live_account_and_schema_conformance(self):
        """Query live API endpoint and assert schema conformance on real production objects."""
        url = "https://api.external-service.com/v2/resources"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "FunctionalTestRunner/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertIsInstance(data, (dict, list))
        # Validate critical contract fields
        resources = data.get("items", []) if isinstance(data, dict) else data
        for item in resources:
            self.assertIn("id", item)
            self.assertIn("status", item)
```

---

## Live Endpoint Health Probing
If active cloud instances or servers are running under the user's account, verify true network connectivity:
```python
    def test_live_instance_health_probe(self):
        """If an instance is actively running on the account, verify its /health probe responds."""
        active_instances = self.fetch_active_instances()
        if not active_instances:
            self.skipTest("No active instances currently running to probe.")

        target = active_instances[0]
        health_url = f"https://{target['id']}.service-domain.net/health"
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body.get("status"), "ok")
```
