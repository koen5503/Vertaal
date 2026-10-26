"""
tests/test_deployment_contract.py

Comprehensive Deployment and FUSE Storage Isolation Contract Verification Suite.
Validates:
1. System dependency declarations for external installers (Contract 1)
2. FUSE storage isolation and safe byte-stream copying (Contract 2)
3. Model caching and re-use verification (Contract 3)
4. Shell script syntax validity (Contract 4)
5. UI dynamic column header mapping in index.html and viewer.html
"""

import os
import re
import subprocess
import unittest


class TestDeploymentContract(unittest.TestCase):
    """Rigorous contract validation for RunPod deployment scripts and UI templates."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.boot_sh_path = os.path.join(cls.project_root, "boot.sh")
        cls.setup_once_path = os.path.join(cls.project_root, "setup_once.sh")
        cls.index_html_path = os.path.join(cls.project_root, "templates", "index.html")
        cls.viewer_html_path = os.path.join(cls.project_root, "templates", "viewer.html")

        with open(cls.boot_sh_path, "r", encoding="utf-8") as f:
            cls.boot_src = f.read()

        with open(cls.setup_once_path, "r", encoding="utf-8") as f:
            cls.setup_src = f.read()

        with open(cls.index_html_path, "r", encoding="utf-8") as f:
            cls.index_src = f.read()

        with open(cls.viewer_html_path, "r", encoding="utf-8") as f:
            cls.viewer_src = f.read()

    # --------------------------------------------------------------------------
    # Contract 1: Systeem-afhankelijkheden
    # --------------------------------------------------------------------------
    def test_contract_1_system_dependencies(self):
        """Verify all tools required by external installers are declared in step [1/8]."""
        required_tools = ["zstd", "curl", "procps", "ffmpeg", "libsndfile1"]

        # 1. setup_once.sh step [1/8]
        step1_setup = re.search(r"=== \[1/8\].*?(?==== \[2/8\])", self.setup_src, re.DOTALL)
        self.assertIsNotNone(step1_setup, "setup_once.sh must contain step [1/8]")
        step1_setup_text = step1_setup.group(0)

        for tool in required_tools:
            self.assertIn(
                tool,
                step1_setup_text,
                f"setup_once.sh step [1/8] missing required tool: {tool}",
            )

        # 2. boot.sh step [1/8]
        step1_boot = re.search(r"=== \[1/8\].*?(?==== \[2/8\])", self.boot_src, re.DOTALL)
        self.assertIsNotNone(step1_boot, "boot.sh must contain step [1/8]")
        step1_boot_text = step1_boot.group(0)

        for tool in required_tools:
            self.assertIn(
                tool,
                step1_boot_text,
                f"boot.sh step [1/8] missing required tool: {tool}",
            )

    # --------------------------------------------------------------------------
    # Contract 2: FUSE Storage Isolation
    # --------------------------------------------------------------------------
    def test_contract_2_fuse_storage_isolation(self):
        """
        Verify scripts adhere to FUSE object storage restrictions:
        - No un-isolated `cp -r ... /workspace`
        - No `chmod` or `chown` calls targeting /workspace
        - No python virtual environments created on /workspace (must be on /root)
        - Safe byte-stream copyfileobj used for model staging
        """
        # Ensure no chmod / chown on workspace
        self.assertNotRegex(
            self.setup_src,
            r"(chmod|chown)\s+.*\/workspace",
            "setup_once.sh must not execute chmod or chown on /workspace",
        )
        self.assertNotRegex(
            self.boot_src,
            r"(chmod|chown)\s+.*\/workspace",
            "boot.sh must not execute chmod or chown on /workspace",
        )

        # Ensure no direct venv creation on /workspace
        self.assertNotIn(
            "python3 -m venv /workspace",
            self.setup_src,
            "setup_once.sh must not create venv on /workspace directly",
        )
        self.assertIn(
            "/root/venv",
            self.setup_src,
            "setup_once.sh must build venv on fast volatile local SSD (/root/venv)",
        )

        # Ensure pure byte-stream copying (copyfileobj) is used, not standard copytree
        self.assertIn(
            "copyfileobj",
            self.setup_src,
            "setup_once.sh must use copyfileobj byte-stream copier for /workspace",
        )
        self.assertIn(
            "copyfileobj",
            self.boot_src,
            "boot.sh must use copyfileobj byte-stream copier for staging Ollama models",
        )
        self.assertNotIn(
            "shutil.copytree",
            self.setup_src,
            "setup_once.sh must not use shutil.copytree due to FUSE copystat chmod restrictions",
        )
        self.assertNotIn(
            "shutil.copytree",
            self.boot_src,
            "boot.sh must not use shutil.copytree due to FUSE copystat chmod restrictions",
        )

    # --------------------------------------------------------------------------
    # Contract 3: Model Re-use Verification
    # --------------------------------------------------------------------------
    def test_contract_3_model_reuse_verification(self):
        """Verify setup_once.sh checks model presence before triggering heavy downloads."""
        # Whisper model check
        self.assertIn(
            '/workspace/models/faster-whisper-large-v3/model.bin"',
            self.setup_src,
            "setup_once.sh must verify model.bin before downloading Faster-Whisper",
        )

        # Ollama blobs check
        self.assertIn(
            "/workspace/models/ollama/blobs",
            self.setup_src,
            "setup_once.sh must verify Ollama blobs before pulling model",
        )

    # --------------------------------------------------------------------------
    # Contract 4: Shell Syntax Verification
    # --------------------------------------------------------------------------
    def test_contract_4_shell_syntax(self):
        """Validate bash syntax for boot.sh and setup_once.sh using `bash -n`."""
        for script_name, script_path in [
            ("boot.sh", self.boot_sh_path),
            ("setup_once.sh", self.setup_once_path),
        ]:
            proc = subprocess.run(
                ["bash", "-n", script_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Bash syntax error in {script_name}:\n{proc.stderr}",
            )

    # --------------------------------------------------------------------------
    # UI Dynamic Column Headers & Language Display Mapping
    # --------------------------------------------------------------------------
    def test_dynamic_column_headers_in_templates(self):
        """Verify dynamic column headers, LANG_DISPLAY mapping, and element IDs in templates."""
        # 1. index.html checks
        self.assertIn("LANG_DISPLAY", self.index_src)
        self.assertIn("getLangLabel", self.index_src)
        self.assertIn("updateColumnHeaders", self.index_src)
        self.assertIn("header-source", self.index_src)
        self.assertIn("header-trans1", self.index_src)
        self.assertIn("header-trans3", self.index_src)
        self.assertIn("header-trans2", self.index_src)
        self.assertIn("header-trans4", self.index_src)

        # 2. viewer.html checks
        self.assertIn("LANG_DISPLAY", self.viewer_src)
        self.assertIn("getLangLabel", self.viewer_src)
        self.assertIn("updateLabels", self.viewer_src)
        self.assertIn("header-source", self.viewer_src)
        self.assertIn("header-trans1", self.viewer_src)
        self.assertIn("header-trans3", self.viewer_src)
        self.assertIn("header-trans2", self.viewer_src)
        self.assertIn("header-trans4", self.viewer_src)
        self.assertIn("label-source", self.viewer_src)
        self.assertIn("label-trans1", self.viewer_src)
        self.assertIn("label-trans3", self.viewer_src)
        self.assertIn("label-trans2", self.viewer_src)
        self.assertIn("label-trans4", self.viewer_src)

    def test_pod_adoption_and_telemetry_enhancements(self):
        """Verify pod adoption logic, removal of podResume, 5s polling, and telemetry timer."""
        runpod_engine_path = os.path.join(self.project_root, "core", "stt", "runpod_engine.py")
        telemetry_html_path = os.path.join(self.project_root, "templates", "telemetry.html")

        with open(runpod_engine_path, "r", encoding="utf-8") as f:
            runpod_src = f.read()

        with open(telemetry_html_path, "r", encoding="utf-8") as f:
            telemetry_src = f.read()

        # 1. Pod adoption check in manage_runpod_deployment
        self.assertIn("running_pods = [", runpod_src)
        self.assertIn("desiredStatus') == 'RUNNING'", runpod_src)
        self.assertIn("Found existing active pod", runpod_src)
        self.assertNotIn("podResume", runpod_src, "Legacy podResume mutation must be removed")

        # 2. Fast 5-second polling loop and elapsed time tracking
        self.assertIn("await asyncio.sleep(5)", runpod_src)
        self.assertIn("elapsed_sec", runpod_src)
        self.assertIn("verlopen", runpod_src)

        # 3. Telemetry boot timer and log accumulation
        self.assertIn('id="boot-timer-text"', telemetry_src)
        self.assertIn("startBootTimer", telemetry_src)
        self.assertIn("appendBootLogs", telemetry_src)
        self.assertIn("accumulatedBootLogs", telemetry_src)


if __name__ == "__main__":
    unittest.main()

