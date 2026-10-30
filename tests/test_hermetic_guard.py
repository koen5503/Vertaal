import os
import sys
import json
import tempfile
import unittest
import subprocess
from pathlib import Path


SITECUSTOMIZE_CODE = """
import os
import sys
import socket
import json
import traceback
import threading

# Ensure urllib / requests do not route outbound traffic through a local proxy
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

LOG_FILE = os.environ.get("HERMETIC_GUARD_LOG_FILE", "")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

_local = threading.local()

def _is_loopback(host):
    if not host:
        return True
    host_str = str(host).lower().strip()
    if host_str in LOOPBACK_HOSTS or host_str.startswith("127."):
        return True
    return False

def _record_and_refuse(host, port, origin_func):
    if getattr(_local, "in_guard", False):
        raise ConnectionRefusedError(f"Hermetic guard refused outbound connection to {host}:{port}")
    _local.in_guard = True
    try:
        call_stack = [line.strip() for line in traceback.format_stack()[:-1]]
        entry = {
            "host": str(host),
            "port": port,
            "origin": origin_func,
            "stack": call_stack
        }
        if LOG_FILE:
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + chr(10))
            except Exception:
                pass
        raise ConnectionRefusedError(f"Hermetic guard refused outbound connection to {host}:{port}")
    finally:
        _local.in_guard = False

_orig_socket_connect = socket.socket.connect
_orig_create_connection = socket.create_connection

def _guarded_connect(self, address):
    host = None
    port = None
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
    elif isinstance(address, str):
        return _orig_socket_connect(self, address)

    if host is not None and not _is_loopback(host):
        _record_and_refuse(host, port, "socket.socket.connect")

    return _orig_socket_connect(self, address)

def _guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
    host = None
    port = None
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]

    if host is not None and not _is_loopback(host):
        _record_and_refuse(host, port, "socket.create_connection")

    return _orig_create_connection(address, timeout=timeout, source_address=source_address, *args, **kwargs)

socket.socket.connect = _guarded_connect
socket.create_connection = _guarded_create_connection
"""


class TestHermeticRegressionGuard(unittest.TestCase):
    """Regression guard enforcing that the hermetic test suite makes 0 outbound non-loopback connections."""

    def test_hermetic_test_suite_makes_no_outbound_network_calls(self):
        tests_dir = Path(__file__).resolve().parent
        project_root = tests_dir.parent

        excluded_files = {
            "test_live_cloud_contract.py",
            "test_live_cloud_e2e_audio_pipeline.py",
            "test_hermetic_guard.py",
        }

        test_files = sorted([
            f.name for f in tests_dir.glob("test_*.py")
            if f.name not in excluded_files
        ])
        self.assertTrue(len(test_files) > 0, "No test files found to verify under hermetic guard")

        module_names = [f"tests.{f[:-3]}" for f in test_files]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_file = temp_path / "outbound_attempts.jsonl"
            sitecustomize_path = temp_path / "sitecustomize.py"

            sitecustomize_path.write_text(SITECUSTOMIZE_CODE, encoding="utf-8")

            # Environment for subprocess
            env = os.environ.copy()
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
                env.pop(k, None)

            env["HERMETIC_GUARD_LOG_FILE"] = str(log_file)
            env["RUNPOD_API_KEY"] = "rpa_TRIPWIRE_DO_NOT_USE"
            env["AUTO_DEPLOY_RUNPOD"] = "true"
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else temp_dir

            # Run all hermetic modules under sitecustomize interceptor
            cmd = [sys.executable, "-m", "unittest"] + module_names
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=180
            )

            # Inspect attempts
            attempts = []
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            attempts.append(json.loads(line))

            failure_msg = ""
            if attempts:
                details = [f"- {a['host']}:{a['port']} (via {a['origin']})" for a in attempts]
                failure_msg = f"Non-loopback connection attempts detected ({len(attempts)}):\n" + "\n".join(details)

            self.assertEqual(len(attempts), 0, failure_msg)
            self.assertEqual(
                proc.returncode, 0,
                f"Subprocess unittest failed with return code {proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
