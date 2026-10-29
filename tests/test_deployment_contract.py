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

        # Ensure pure byte-stream copying (copyfileobj) is used in setup_once.sh, but not in boot.sh
        self.assertIn(
            "copyfileobj",
            self.setup_src,
            "setup_once.sh must use copyfileobj byte-stream copier for /workspace",
        )
        self.assertNotIn(
            "copyfileobj",
            self.boot_src,
            "boot.sh must not run copyfileobj loops to copy multi-GB models on boot",
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

        # Ollama binary bundle caching & restore check (supporting .tar.zst alongside .tar)
        self.assertTrue(
            "/workspace/ollama_bundle.tar.zst" in self.setup_src or "/workspace/ollama_bundle.tar" in self.setup_src,
            "setup_once.sh must cache Ollama binaries to /workspace/ollama_bundle.tar.zst or .tar",
        )
        self.assertTrue(
            "/workspace/ollama_bundle.tar.zst" in self.boot_src or "/workspace/ollama_bundle.tar" in self.boot_src,
            "boot.sh must restore Ollama binaries from /workspace/ollama_bundle.tar.zst or .tar",
        )
        self.assertTrue(
            '[ -s "/workspace/ollama_bundle.tar.zst" ]' in self.boot_src or '[ -s "/workspace/ollama_bundle.tar" ]' in self.boot_src,
            "boot.sh must verify that ollama bundle is non-empty before extracting",
        )
        self.assertTrue(
            "/tmp/ollama_bundle.tar.zst" in self.setup_src or "/tmp/ollama_bundle.tar" in self.setup_src,
            "setup_once.sh must stage ollama bundle via /tmp to prevent partial writes",
        )
        self.assertTrue(
            "/tmp/ollama_bundle.tar.zst" in self.boot_src or "/tmp/ollama_bundle.tar" in self.boot_src,
            "boot.sh must stage ollama bundle via /tmp to prevent partial writes",
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
        self.assertIn("connIndicator.innerText = '⏳ ' + data.stage;", telemetry_src)

        # 4. Streamed log deque buffer and removal of artificial 40-line cutoff
        self.assertIn("clean_lines = deque(maxlen=200)", runpod_src)
        self.assertNotIn("len(clean_lines) >= 40", runpod_src)

        # 5. Verify condense_runpod_logs parses latest step past 100 lines and filters curl noise
        from core.stt.runpod_engine import condense_runpod_logs
        mock_logs = "\n".join([f"line {i}: apt-get output" for i in range(80)])
        mock_logs += "\n=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ===\n"
        mock_logs += "##O#- # 100.0%\n"
        mock_logs += "### 99.9%\n"
        mock_logs += "Restoring Ollama engine from Global Volume cache (1-2s)...\n"

        stage, recent_lines = condense_runpod_logs(mock_logs)
        self.assertIn("[5/8]", stage)
        self.assertIn("Ollama binaries herstellen", stage)
        self.assertNotIn("### 99.9%", recent_lines)

    def test_runpod_faster_whisper_logging_and_accuracy_configuration(self):
        """Verify Faster-Whisper maintains full accuracy parameters and latency timing logs in runpod_server.py."""
        runpod_server_path = os.path.join(self.project_root, "runpod_server.py")
        with open(runpod_server_path, "r", encoding="utf-8") as f:
            server_src = f.read()

        # Whisper maintains full accuracy (standard beam search and prompt context preserved)
        self.assertIn('"task": "transcribe"', server_src)
        self.assertIn('transcribe_args["initial_prompt"] = last_transcript', server_src)
        self.assertIn('"temperature": 0.0', server_src)
        self.assertIn('"condition_on_previous_text": False', server_src)

        # STT and LLM latency timing logs
        self.assertIn("[Whisper] Transcribed", server_src)
        self.assertIn("[Ollama] Translated", server_src)

        # Translation debouncing, chain_busy skip, and back-translation gating
        self.assertIn("active_chain_tasks", server_src)
        self.assertIn("chain_busy", server_src)
        self.assertIn('segment_status == "final"', server_src)
        self.assertIn("export OLLAMA_NUM_PARALLEL=2", self.boot_src)
        self.assertIn("Pre-warming Ollama aya-expanse:8b in GPU VRAM", self.boot_src)
        self.assertIn('ollama run aya-expanse:8b "hi"', self.boot_src)
        self.assertRegex(
            self.boot_src,
            r'\(ollama run aya-expanse:8b "hi".*?\)\s*&',
            "boot.sh must pre-warm Ollama aya-expanse:8b non-blocking in background (&) for concurrent VRAM loading",
        )

    def test_whisper_nvme_staging_and_path_resolution(self):
        """Verify Faster-Whisper is linked to /root/models in boot.sh via ln -sfn and resolved with local_files_only."""
        # 1. boot.sh checks
        self.assertIn("/root/models", self.boot_src)
        self.assertIn("/root/models/faster-whisper-large-v3", self.boot_src)
        self.assertIn("Linking Faster-Whisper large-v3 from /workspace to local SSD", self.boot_src)
        self.assertIn("ln -sfn /workspace/models/faster-whisper-large-v3 /root/models/faster-whisper-large-v3", self.boot_src)
        self.assertIn("export WHISPER_MODEL=/root/models/faster-whisper-large-v3", self.boot_src)

        # 2. runpod_server.py checks
        runpod_server_path = os.path.join(self.project_root, "runpod_server.py")
        with open(runpod_server_path, "r", encoding="utf-8") as f:
            server_src = f.read()

        self.assertIn("resolve_whisper_model_path", server_src)
        self.assertIn("/root/models", server_src)
        self.assertIn("/workspace/models", server_src)
        self.assertIn("local_files_only=True", server_src)
        self.assertIn("[Whisper Init] Loading faster-whisper from local directory", server_src)
        self.assertIn("loaded into GPU in", server_src)

        # 3. Dynamic unit test of resolve_whisper_model_path logic
        import tempfile
        from unittest.mock import patch

        # Import or extract resolve_whisper_model_path
        namespace = {}
        exec(compile(
            """import os\nfrom typing import Optional\n""" +
            server_src[server_src.find("def resolve_whisper_model_path"):server_src.find("def load_whisper_model")],
            "runpod_server.py", "exec"
        ), namespace)
        resolve_func = namespace["resolve_whisper_model_path"]

        with tempfile.TemporaryDirectory() as tmp_root, tempfile.TemporaryDirectory() as tmp_ws:
            root_model = os.path.join(tmp_root, "faster-whisper-large-v3")
            ws_model = os.path.join(tmp_ws, "faster-whisper-large-v3")
            os.makedirs(root_model, exist_ok=True)
            os.makedirs(ws_model, exist_ok=True)

            with open(os.path.join(root_model, "model.bin"), "wb") as f:
                f.write(b"mock_model")
            with open(os.path.join(ws_model, "model.bin"), "wb") as f:
                f.write(b"mock_model")

            # Patch os.path.join to redirect /root/models and /workspace/models
            orig_join = os.path.join
            def mock_join(*args):
                if len(args) >= 2:
                    if args[0] == "/root/models":
                        return orig_join(tmp_root, *args[1:])
                    elif args[0] == "/workspace/models":
                        return orig_join(tmp_ws, *args[1:])
                return orig_join(*args)

            with patch("os.path.join", side_effect=mock_join):
                res = resolve_func("large-v3")
                self.assertIsNotNone(res)
                self.assertTrue(res.startswith(tmp_root), f"Expected /root/models preference, got {res}")

    def test_runpod_global_volume_reliability_risks(self):
        """
        Verify fixes for the 3 RunPod Global Volume Reliability Risks:
        1. Unguarded GitHub Raw CDN dependency on boot (guarded via /tmp, fallback URL, and reuse of existing files)
        2. Partial tarball corruption risk on /workspace/venv_bundle.tar (staged via /tmp, self-healing in boot.sh)
        3. Instant symlinks on /root for both Ollama and Faster-Whisper (ln -sfn directly to /workspace/models)
        """
        # Risk 1: GitHub Raw CDN Dependency & Staging via /tmp
        self.assertIn('REPO_BRANCH="${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}"', self.boot_src)
        self.assertIn('REPO_RAW="${RUNPOD_REPO_RAW:-https://raw.githubusercontent.com/koen5503/Vertaal/${REPO_BRANCH}}"', self.boot_src)
        self.assertIn('FALLBACK_REPO_RAW="https://raw.githubusercontent.com/koen5503/Vertaal/main"', self.boot_src)
        self.assertIn("sync_repo_file", self.boot_src)
        self.assertIn("/tmp/", self.boot_src)
        self.assertIn('sync_repo_file "setup_once.sh" "/workspace/setup_once.sh"', self.boot_src)
        self.assertIn('sync_repo_file "runpod_server.py" "/workspace/app/runpod_server.py"', self.boot_src)
        self.assertIn('[ -s "$dest_path" ]', self.boot_src)

        # Risk 2: venv_bundle atomic creation and self-healing (supporting .tar.zst alongside .tar)
        # setup_once.sh step 7 creates in /tmp first and checks non-empty
        self.assertTrue(
            'tar -I "zstd -1 -T0" -cf /tmp/venv_bundle.tar.zst -C /root venv' in self.setup_src
            or "tar -cf /tmp/venv_bundle.tar -C /root venv" in self.setup_src,
            "setup_once.sh must archive venv via /tmp",
        )
        self.assertTrue(
            '[ -s "/tmp/venv_bundle.tar.zst" ]' in self.setup_src or '[ -s "/tmp/venv_bundle.tar" ]' in self.setup_src,
            "setup_once.sh must verify non-empty venv archive in /tmp",
        )
        self.assertTrue(
            "cp -f /tmp/venv_bundle.tar.zst /workspace/venv_bundle.tar.zst" in self.setup_src
            or "cp -f /tmp/venv_bundle.tar /workspace/venv_bundle.tar" in self.setup_src,
            "setup_once.sh must copy venv bundle to /workspace",
        )

        # boot.sh step 2 checks for presence of venv_bundle (zst or tar)
        self.assertTrue(
            '[ ! -s "/workspace/venv_bundle.tar.zst" ]' in self.boot_src or '[ ! -s "/workspace/venv_bundle.tar" ]' in self.boot_src,
            "boot.sh must check for presence of venv bundle on /workspace",
        )
        # boot.sh step 5 self-heals by running setup_once.sh if extraction fails or python binary is missing
        step5_boot = re.search(r"=== \[5/8\].*?(?==== \[6/8\])", self.boot_src, re.DOTALL)
        self.assertIsNotNone(step5_boot)
        step5_text = step5_boot.group(0)
        self.assertIn("setup_once.sh", step5_text)
        self.assertTrue(
            "rm -f /workspace/venv_bundle.tar" in step5_text or "rm -f /workspace/venv_bundle.tar.zst" in step5_text,
            "boot.sh must clean up corrupt venv bundle",
        )
        self.assertIn("/root/venv/bin/python3", step5_text)

        # Risk 3: Instant symlinks on /root for both Ollama and Faster-Whisper
        self.assertIn("mkdir -p /root/app /tmp/hf_cache /tmp/torch_cache /root/.ollama /root/models", self.boot_src)
        self.assertIn("ln -sfn /workspace/models/ollama /root/.ollama/models", self.boot_src)
        self.assertIn("ln -sfn /workspace/models/faster-whisper-large-v3 /root/models/faster-whisper-large-v3", self.boot_src)
        self.assertIn("export OLLAMA_MODELS=/root/.ollama/models", self.boot_src)
        self.assertIn("export WHISPER_MODEL=/root/models/faster-whisper-large-v3", self.boot_src)

    def test_bundle_version_manifest_and_python_coupling_guard(self):
        """
        Verify Bundle Version Manifest and Python Version Coupling Guard:
        1. BUNDLE_VERSION='2026.10.1' defined in both boot.sh and setup_once.sh
        2. setup_once.sh writes /workspace/.bundle_version upon successful archiving
        3. boot.sh checks /workspace/.bundle_version and triggers upgrade on mismatch
        4. boot.sh checks container OS vs venv python minor version and rebuilds venv if mismatched
        """
        # 1. BUNDLE_VERSION constant
        self.assertIn('BUNDLE_VERSION="2026.10.1"', self.boot_src)
        self.assertIn('BUNDLE_VERSION="2026.10.1"', self.setup_src)

        # 2. setup_once.sh manifest write
        self.assertIn('echo "$BUNDLE_VERSION" > /workspace/.bundle_version', self.setup_src)

        # 3. boot.sh bundle version check & upgrade trigger
        self.assertIn("/workspace/.bundle_version", self.boot_src)
        self.assertIn("Bundle version mismatch", self.boot_src)
        self.assertIn("Triggering automated upgrade", self.boot_src)
        self.assertIn("rm -f /workspace/venv_bundle.tar.zst /workspace/venv_bundle.tar", self.boot_src)

        # 4. boot.sh python minor version guard
        self.assertIn("SYS_PY_VER=$(python3 -c", self.boot_src)
        self.assertIn("sys.version_info.major", self.boot_src)
        self.assertIn("sys.version_info.minor", self.boot_src)
        self.assertIn("VENV_PY_VER=", self.boot_src)
        self.assertIn('if [ -n "$SYS_PY_VER" ] && [ -n "$VENV_PY_VER" ] && [ "$SYS_PY_VER" != "$VENV_PY_VER" ]; then', self.boot_src)
        self.assertIn("Incompatible Python minor version detected", self.boot_src)
        self.assertIn("rm -rf /root/venv /workspace/venv_bundle.tar.zst /workspace/venv_bundle.tar /workspace/.bundle_version", self.boot_src)

    def test_dynamic_repository_branch_and_boot_cmd(self):
        """
        Verify dynamic branch configuration and RUNPOD_BOOT_CMD:
        1. RUNPOD_REPO_BRANCH in boot.sh and setup_once.sh defaulting to feature branch
        2. Fallback to main branch, skipping duplicate curl if REPO_BRANCH == 'main'
        3. RUNPOD_BOOT_CMD in runpod_engine.py supports RUNPOD_REPO_BRANCH and main fallback
        """
        from core.stt.runpod_engine import RUNPOD_BOOT_CMD, RUNPOD_OVERRIDE_CMD

        self.assertIn('REPO_BRANCH="${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}"', self.boot_src)
        self.assertIn('REPO_BRANCH="${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}"', self.setup_src)
        self.assertIn('[ "$REPO_BRANCH" != "main" ]', self.boot_src)
        self.assertIn('[ "$REPO_BRANCH" != "main" ]', self.setup_src)

        self.assertIn("bash -c", RUNPOD_BOOT_CMD)
        self.assertIn("${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}", RUNPOD_BOOT_CMD)
        self.assertIn("koen5503/Vertaal/main/boot.sh", RUNPOD_BOOT_CMD)
        self.assertEqual(RUNPOD_OVERRIDE_CMD, RUNPOD_BOOT_CMD)

    def test_ensure_template_http_ports_rest_v2_and_graphql_paths(self):
        """
        Verify ensure_template_http_ports handles REST API v2 primary and GraphQL fallback:
        - REST API v2 primary success with 8000/http (no GraphQL calls)
        - REST API v2 404 falling back gracefully to GraphQL query + mutation
        - Exception safety: network/parsing failure never raises unhandled exception
        """
        import json
        import urllib.error
        from unittest.mock import MagicMock, patch
        from core.stt.runpod_engine import RunPodSTTEngine

        mock_engine = MagicMock()
        stt = RunPodSTTEngine(engine=mock_engine)

        # 1. REST API v2 primary success path (returns 8000/http directly)
        mock_v2_resp = MagicMock()
        mock_v2_resp.status = 200
        mock_v2_resp.headers = {}
        mock_v2_resp.read.return_value = json.dumps({
            "template": {
                "id": "tmpl-v2-ok",
                "ports": "8000/http",
                "name": "ondertitels-v2"
            }
        }).encode("utf-8")
        mock_v2_resp.__enter__.return_value = mock_v2_resp

        with patch("urllib.request.urlopen", return_value=mock_v2_resp) as mock_urlopen:
            success = stt.ensure_template_http_ports("test-key", "tmpl-v2-ok")
            self.assertTrue(success)
            self.assertEqual(mock_urlopen.call_count, 1)

        # 2. REST API v2 404 fallback to GraphQL query + mutation
        gql_query_resp = {
            "data": {
                "myself": {
                    "podTemplates": [{
                        "id": "tmpl-fallback-gql",
                        "ports": "8000/tcp",
                        "name": "ondertitels-gql",
                        "imageName": "runpod/pytorch:2.1.0",
                        "containerDiskInGb": 20,
                        "volumeInGb": 20,
                        "volumeMountPath": "/workspace",
                        "dockerArgs": "",
                        "env": []
                    }]
                }
            }
        }
        gql_mut_resp = {
            "data": {
                "saveTemplate": {
                    "id": "tmpl-fallback-gql",
                    "ports": "8000/http"
                }
            }
        }

        routed_urls = []
        def router(req, timeout=10.0):
            routed_urls.append(req.full_url)
            if "/v2/templates" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            if "graphql" in req.full_url:
                body = json.loads(req.data.decode("utf-8"))
                mock_r = MagicMock()
                mock_r.status = 200
                mock_r.headers = {}
                if "podTemplates" in body["query"]:
                    mock_r.read.return_value = json.dumps(gql_query_resp).encode("utf-8")
                else:
                    mock_r.read.return_value = json.dumps(gql_mut_resp).encode("utf-8")
                mock_r.__enter__.return_value = mock_r
                return mock_r
            raise RuntimeError(f"Unexpected URL: {req.full_url}")

        with patch("urllib.request.urlopen", side_effect=router):
            success = stt.ensure_template_http_ports("test-key", "tmpl-fallback-gql")
            self.assertTrue(success)
            # Must have attempted REST v2 first, then GraphQL query and mutation
            self.assertTrue(any("/v2/templates" in u for u in routed_urls))
            self.assertTrue(any("graphql" in u for u in routed_urls))

        # 3. Exception guard: unexpected error never throws
        def crashing_urlopen(req, timeout=10.0):
            raise OSError("Network connection lost")

        with patch("urllib.request.urlopen", side_effect=crashing_urlopen):
            # Must return False gracefully without raising
            success = stt.ensure_template_http_ports("test-key", "tmpl-crash")
            self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()



