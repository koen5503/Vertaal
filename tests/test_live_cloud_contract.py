"""Live cloud API contract smoke tests for RunPod REST API v2 and proxy health.

Strictly follows Pillar 2 of .agents/skills/functional-regression-testing/SKILL.md
and references/live_api_contract_guide.md.
Gated on RUNPOD_API_KEY environment variable. Zero execution in offline CI / dev environments.
"""

import json
import os
import unittest
import urllib.request
from typing import List, Dict, Any

from core.stt.runpod_engine import RunPodSTTEngine


def _load_env_if_present():
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except Exception:
            pass

_load_env_if_present()


@unittest.skipUnless(os.getenv("RUNPOD_API_KEY"), "Skipping live cloud contract: RUNPOD_API_KEY not set")
class TestLiveCloudContract(unittest.TestCase):
    """Verifies live external cloud contract conformance and network routability against RunPod API v2."""

    def setUp(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.engine = RunPodSTTEngine()

    def test_live_runpod_v2_pods_schema_conformance(self):
        """Query live RunPod REST API v2 and assert schema conformance across all user pods."""
        pods = self.engine._fetch_pods_list(self.api_key)
        self.assertIsInstance(pods, list, f"Expected list of pods from _fetch_pods_list, got {type(pods)}")

        for pod in pods:
            self.assertIsInstance(pod, dict, f"Each pod must be a dict, got {type(pod)}")
            # id
            self.assertIn("id", pod)
            self.assertIsInstance(pod["id"], str)
            self.assertTrue(len(pod["id"]) > 0, "Pod id must not be empty")

            # name
            self.assertIn("name", pod)
            self.assertIsInstance(pod["name"], str)

            # desiredStatus
            self.assertIn("desiredStatus", pod)
            self.assertIsInstance(pod["desiredStatus"], str)
            self.assertIn(pod["desiredStatus"], ["RUNNING", "STOPPED", "EXITED", "TERMINATING", "PAUSED", "RESTARTING", "INITIALIZING"])

            # runtime.ports
            self.assertIn("runtime", pod)
            self.assertIsInstance(pod["runtime"], dict)
            self.assertIn("ports", pod["runtime"])
            self.assertIsInstance(pod["runtime"]["ports"], list)
            for p in pod["runtime"]["ports"]:
                self.assertIsInstance(p, dict)
                self.assertIn("privatePort", p)

            # proxy_url
            self.assertIn("proxy_url", pod)
            self.assertIsInstance(pod["proxy_url"], str)
            self.assertTrue(
                pod["proxy_url"].startswith("wss://") and ".proxy.runpod.net" in pod["proxy_url"],
                f"proxy_url must be valid RunPod proxy format, got: {pod['proxy_url']}"
            )

    def test_live_pod_health_probe_if_running(self):
        """If any pod on the account is actively RUNNING, probe https://<pod_id>-8000.proxy.runpod.net/health."""
        pods = self.engine._fetch_pods_list(self.api_key)
        running_pods = [p for p in pods if p.get("desiredStatus") == "RUNNING"]

        if not running_pods:
            self.skipTest("No active pod currently running to probe.")

        target_pod = running_pods[0]
        pod_id = target_pod["id"]
        health_url = f"https://{pod_id}-8000.proxy.runpod.net/health"

        req = urllib.request.Request(
            health_url,
            headers={
                "User-Agent": "Ondertitels-FunctionalTest/1.0",
                "Accept": "application/json",
            }
        )

        with urllib.request.urlopen(req, timeout=10.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertTrue(data.get("whisper_ready") is True)


if __name__ == "__main__":
    unittest.main()
