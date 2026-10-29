---
name: functional-regression-testing
description: >-
  Build, run, and maintain multi-tiered functional regression test suites that eliminate mock-drift,
  verify live external API contracts, test end-to-end pipeline execution over real loopback sockets,
  and enforce measurable branch and line code coverage. Activate this skill when the user asks to
  create regression tests, eliminate synthetic mocks, measure test coverage, or verify
  bug fixes against real production contracts.
---

# Multi-Tiered Functional Regression Testing & Coverage Skill

> [!IMPORTANT]
> **DEFINITION: A FUNCTIONAL TEST IS A REAL FUNCTIONAL END-TO-END TEST**  
> In all current and future projects, a "functional test" is **strictly defined as a genuine, real-world end-to-end pipeline test**. It must never be reduced to local loopback dummies or synthetic stubs when real compute, cloud infrastructure, or external APIs are involved.  
> 
> A True Functional End-to-End Test MUST:
> 1. **Engage Real Target Infrastructure**: Connect to or provision the authentic production/staging backend (e.g., actual cloud GPU pod, live database, real server).
> 2. **Ingest Real Domain Media / Payloads**: Use genuine input files (e.g., an authentic `.wav` speech recording specified in `.env`, validating its existence and format first), never empty dummy bytes or pure zero-streams.
> 3. **Verify Actual End-to-End Computation**: Assert observable business outcomes (e.g., actual words transcribed by the speech model, actual translations produced by the language model).
> 4. **Guarantee Autonomous Teardown**: Automatically terminate or stop any provisioned cloud resources upon completion to leave zero orphaned costs.

---

## The Four Pillars of Functional Regression Testing

```mermaid
flowchart TD
    subgraph OfflineTiers ["Offline & CI Safe (Millisecond Execution)"]
        Tier1["Pillar 1: Golden Fixture Contract Tests<br/>(Raw authentic JSON/SSE dumps, zero synthetic mocks)"]
        Tier3["Pillar 3: Ephemeral Loopback Pipeline Tests<br/>(Real server + client on loopback port, real streams)"]
    end
    subgraph LiveTier ["Conditional / Opt-In (Environment Gated)"]
        Tier2["Pillar 2: Live API Contract Smoke Tests<br/>(Opt-in via API Key, tests real external cloud endpoints)"]
    end
    subgraph QualityGate ["Verification & Coverage Gate"]
        Tier4["Pillar 4: Quality Gate & Branch Coverage<br/>(100% pass rate, branch coverage >= 85%, zero regressions)"]
    end

    Tier1 --> Tier4
    Tier3 --> Tier4
    Tier2 -.->|When API key present| Tier4
```

---

## Pillar 1: Golden Production Fixtures (Anti-Mock Contract Testing)

### Why Unit Mocks Fail
When engineers write unit mocks (e.g. `mock_resp.json.return_value = {"ports": [{"privatePort": 8000}]}`), they test their **mental model** of the third-party service, not reality. When the real API returns `{"ports": ["8000/http"], "runtime": null}`, unit tests remain 100% green while production crashes with an `AttributeError` or `KeyError`.

### General Implementation Pattern
1. **Directory Structure**: Store verbatim snapshots in `tests/fixtures/<service>_<endpoint>_<scenario>.json`:
   - `tests/fixtures/service_v2_resource_active.json`
   - `tests/fixtures/service_v2_resource_empty.json`
   - `tests/fixtures/service_v2_resource_edge_types.json`
2. **Defensive Type Permutations**: Test parsers against:
   - Scalar string vs. list of strings vs. list of dictionaries.
   - `null` / `None` values where objects or lists were expected.
   - Unexpected extra fields and missing optional keys.
3. **Fixture Ingestion Test**:
   ```python
   def test_golden_contract_ingestion(self):
       with open("tests/fixtures/service_v2_resource_active.json", "r", encoding="utf-8") as f:
           raw_payload = json.load(f)
       # Feed directly into production parser / client
       result = client.parse_resource(raw_payload)
       self.assertIsNotNone(result.endpoint_url)
       self.assertEqual(result.status, "ACTIVE")
   ```

---

## Pillar 2: Live API Contract Smoke Tests (Opt-In / Conditional)

### Purpose
Detects breaking changes in upstream third-party APIs (e.g., deprecated fields, changed response schemas, rate-limit adjustments) before code reaches end users.

### General Implementation Pattern
- **Conditional Execution**: Gate tests on the presence of the live environment variable or API key so that local development and offline CI runs pass without requiring credentials or incurring cloud costs:
  - **Python (`unittest`)**: `@unittest.skipUnless(os.getenv("<SERVICE>_API_KEY"), "Live <SERVICE>_API_KEY required")`
  - **Python (`pytest`)**: `@pytest.mark.skipif(not os.getenv("<SERVICE>_API_KEY"), reason="...")`
  - **JavaScript (`Jest`)**: `const describeIfLive = process.env.API_KEY ? describe : describe.skip;`
  - **Go**: `if os.Getenv("API_KEY") == "" { t.Skip("Skipping live test") }`
- **What to Verify in Live Smoke Tests**:
  1. **Schema Integrity**: Query the live endpoint (e.g. `GET /v2/resources`) and assert that all required keys are present and have expected types.
  2. **Rate-Limit Compliance**: Verify that headers (`Retry-After`, `X-RateLimit-Remaining`) are parsed correctly.
  3. **Live Health Probe**: If active resources exist on the account, send a lightweight `GET /health` probe to verify end-to-end network routability.

```python
@unittest.skipUnless(os.getenv("SERVICE_API_KEY"), "Live SERVICE_API_KEY required for smoke test")
class TestLiveAPIContract(unittest.TestCase):
    def test_live_schema_and_endpoint_health(self):
        api_key = os.getenv("SERVICE_API_KEY")
        client = ServiceClient(api_key=api_key)
        resources = client.fetch_resources()
        self.assertIsInstance(resources, list)
        for res in resources:
            self.assertTrue(hasattr(res, "id") and len(res.id) > 0)
```

---

## Pillar 3: Real Functional End-to-End Pipeline Testing (Genuine Cloud & Domain Media)

### Purpose
Executes the **complete, live end-to-end pipeline** against actual target infrastructure (e.g., real cloud GPU instance) using authentic domain media (e.g., a real `.wav` speech recording specified in `.env`), verifying that actual business computation (STT transcription and LLM translation) occurs and that cloud resources are autonomously cleaned up.

### Requirements for a Real Functional Pipeline Test:
1. **Target Infrastructure Engagement**:
   - Resolve or rent an authentic cloud instance (e.g. via RunPod REST API v2).
   - Follow boot logs until the server is confirmed listening and models are loaded (`/health` returns HTTP 200).
2. **Authentic Domain Media Verification**:
   - Inspect `.env` for the configured audio/input file (e.g., `DEMO_FILE_PATH=Preek260412.wav`).
   - Verify file existence, non-zero file size, and correct format (`.wav`, 16kHz PCM audio) before starting the stream.
3. **Live Streaming over Authentic Network Route**:
   - Connect the real client over the real public endpoint (e.g., `wss://<pod_id>-8000.proxy.runpod.net/stream`).
   - Stream actual PCM audio chunks sliced from the `.wav` file into the WebSocket.
4. **Observable Business Output Verification**:
   - Assert that **real Speech-to-Text (STT)** transcription events (`col1_text`) are returned with recognized words matching the spoken audio.
   - Assert that **real downstream translation** events (`col2_text` / `trans1`) are produced by the translation engine.
5. **Guaranteed Autonomous Teardown**:
   - The test must automatically terminate or stop the provisioned cloud pod in `tearDown` / `finally` blocks to guarantee zero lingering cloud costs.

*(Note: When live cloud credentials are absent, an in-process loopback daemon on `127.0.0.1:<port>` is used solely as an offline fallback tier, but does NOT substitute for the genuine functional test).*

---

## Pillar 4: Pre-Flight Verification Gate & Coverage Enforcement

### Purpose
Provides a quantitative guarantee that all critical logic, error recovery branches, and edge-cases are actively exercised before merging or deploying.

### General Implementation Pattern
1. **Measure Branch & Line Coverage**:
   Use standard coverage tools configured to track branches:
   - **Python**: `python3 -m coverage run --branch -m unittest discover -s tests -p "test_*.py" -v`
   - **Node.js**: `jest --coverage --coverageReporters="text" --coverageThreshold='{"global":{"branches":85}}'`
   - **Go**: `go test -coverprofile=coverage.out ./... && go tool cover -func=coverage.out`
2. **Coverage Thresholds**:
   - **Statement Coverage**: >= 90% on business and networking logic.
   - **Branch Coverage**: >= 85% across all conditional evaluations (`if/else`, `try/except`, `while`).
   - **Regression Coverage**: 100% of previously resolved production defects must have an explicit, reproducing test.
3. **Inspecting Missing Branches**:
   Review the coverage report's missing line annotations:
   - Ensure rate-limit backoff paths (`HTTP 429`), network disconnection handlers, and fallback routes are tested.

---

## Standard Runbook for New Features & Bug Fixes

When implementing new features or fixing defects:

1. **Bug Reproduction**: Create a failing test in the appropriate tier reproducing the defect using the exact payload or event sequence that triggered the bug.
2. **Fixture Capture**: If the bug was caused by an unexpected external API payload, store the authentic payload in `tests/fixtures/`.
3. **Implementation**: Fix the defect in code with defensive type handling.
4. **Verification Gate**: Execute the full test suite and coverage runner:
   ```bash
   python3 .agents/skills/functional-regression-testing/scripts/run_coverage.py 85
   ```
5. **Confirmation**: Ensure 100% of tests pass across all tiers with zero regressions.

---

## 5. Autonomous Agent Enforcement (Zero User Burden)

This skill is designed specifically to relieve the human user of **all** testing and maintenance burdens:

1. **Zero User Execution**:
   - The AI agent MUST NEVER ask the user to run `unittest`, `pytest`, coverage scripts, or test commands.
   - The AI agent autonomously executes the test runner via tool calls before reporting completion. A task is strictly not done until the agent observes 100% passing tests.
2. **Zero User Maintenance**:
   - The user never writes mocks, asserts schemas, or edits test files.
   - Whenever the agent modifies an API client, refactors a parser, or resolves a production defect, the agent **autonomously captures the new fixture, updates the corresponding tests, and verifies coverage** in the exact same commit.
3. **Self-Healing Test Suites**:
   - When external API contracts evolve, the agent is responsible for capturing the new production payload into `tests/fixtures/` and adapting the contract tests so the test suite maintains itself over time.

