# AGENT TESTING & VERIFICATION PROTOCOL (MANDATORY)

## 1. No Routine Verification Burden on the User
- The AI agent MUST NEVER ask the user to check console logs, inspect container boots, or diagnose runtime errors that the agent can observe itself.
- Whenever a cloud pod is deployed or backend code is altered, the agent MUST autonomously query the RunPod API (`/v2/pods/{id}/logs` and GraphQL) to inspect the container output and verify connectivity.
- A task is ONLY considered done when the agent has verified that the server is listening, models are loaded, and the WebSocket accepts streams.

## 2. Real-World Integration Tests (Not Just Unit Tests)
- Unit tests that test isolated functions (like string trimming or math) are INSUFFICIENT for boot scripts, log parsers, and deployment flows.
- Every change to `boot.sh`, `setup_once.sh`, `runpod_server.py`, or `runpod_engine.py` MUST be accompanied by real-world simulation tests in `tests/`:
  1. **Log Stream Simulation**: Feed `condense_runpod_logs()` and `fetch_pod_logs()` with 500+ lines of realistic, dirty container output (apt-get installs, curl download progress hashes, SSE format, broken JSON, system restarts) to guarantee no premature cutoff or silent failures.
  2. **Error Pattern Escalation**: Fatal keywords (`ERROR:`, `fatal:`, `Traceback`, `Permission denied`, `No such file`, `FAILED`) in container logs MUST be immediately detected and escalated to `ERROR` status badge with the exact line in telemetry, never masked by generic "Initialiseren...".
  3. **Phase & Step Tracking**: Verify that each phase `[1/8]` through `[8/8]` is uniquely detected and tracked.
  4. **FUSE & Memory Constraints**: Verify that no script writes runtime caches or unpacks virtualenvs on `/workspace` (`TMPDIR`, `HF_HOME`, `TORCH_HOME` must point to `/tmp`, and `venv` must unpack to `/root/venv`). Model downloads in `setup_once.sh` MUST stage on local NVMe (`/tmp` or `/root`) before copying flat to `/workspace/models/` via `shutil.copyfileobj`. Conversely, `boot.sh` MUST NOT copy multi-GB models from `/workspace/models/` to `/root` via Python on boot; both Ollama and Faster-Whisper must be symlinked on `/root` (`ln -sfn`) and loaded directly into GPU VRAM.

## 3. Pre-Flight Verification Gate
- Before reporting completion to the user, run:
  `python3 -m unittest discover -s tests -p "test_*.py" -v`
- All tests (both unit and real-world simulation) must pass at 100%.

## 4. Strikt Rolbehoud: Architect vs. Coding Agent
- De primaire agent opereert UITSLUITEND als Lead Architect en Test-Orchestrator.
- De primaire agent mag NOOIT zelfstandig broncode, boot-scripts of tests bewerken via `edit_file` of `write_file`.
- ELKE codewijziging, testcreatie of bugfix MOET verplicht worden gedelegeerd aan de `code_developer_agent` via `invoke_subagent`.
- De taak van de architect is: eisen analyseren, specificaties opstellen, de developer aansturen en de testresultaten toetsen.

## 5. Falen van E2E-testen: Verplichte Stop & Failure-Analyse (Geen Zelfstandige Fixes)
- Als een live end-to-end test FAALT (ongeacht of unit tests slagen), mag de coding agent NOOIT zelfstandig broncode, boot-scripts of tests gaan bewerken om het op te lossen.
- De coding agent MOET onmiddellijk:
  1. De cloud pod beëindigen via REST API v2 (geen verweesde kosten).
  2. Stoppen met uitvoeren.
  3. Een volledig en ongefilterd storingsrapport uitbrengen aan de architect met de exacte foutmelding, traceback en de laatste 50 regels containerlogs.
- Pas nadat de architect en de gebruiker de oorzaakanalyse hebben goedgekeurd en een expliciete herstelopdracht hebben gegeven, mag er code worden gewijzigd.

## 6. Verbod op Zelfstandig Pushen bij Tussentijdse Mislukkingen
- De coding agent mag NOOIT commits pushen naar Git als een tussenstap heeft gefaald of als er nog geen schone 100% lokale pre-flight testrun is behaald over álle testsuites.
- Elke vorm van verhulling of selectieve rapportage van testresultaten is ten strengste verboden.


