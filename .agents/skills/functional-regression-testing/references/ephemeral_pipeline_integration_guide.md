# Ephemeral Pipeline Integration Testing Blueprint

## Overview
Ephemeral pipeline tests verify asynchronous client-server protocols (WebSocket, gRPC, HTTP streaming, SSE) over real TCP loopback sockets (`127.0.0.1`), eliminating synthetic mock drift in transport logic.

---

## Pattern: Loopback Server Daemon with Client Probe

### 1. Ephemeral Port Allocation
To avoid port collisions when tests run concurrently, either bind to port `0` (letting the OS assign a random free port) or use a high isolated test port.

```python
import socket

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
```

### 2. In-Process Server Lifecycle
Start the server in a background thread or daemon subprocess:

```python
import threading
import time
import uvicorn
import unittest

class TestWebSocketPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        config = uvicorn.Config("server_module:app", host="127.0.0.1", port=cls.port, log_level="warning")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()

        # Wait until server is listening
        start = time.time()
        while not cls.server.started and time.time() - start < 5.0:
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=3.0)

    def test_real_client_streaming_handshake(self):
        client = RealStreamingClient(f"ws://127.0.0.1:{self.port}/stream")
        client.connect(timeout=3.0)
        client.send_chunk(b"\x00" * 3200)
        events = client.collect_events(timeout=3.0)
        self.assertTrue(len(events) > 0)
        client.close()
```

---

## Key Verification Checklist
1. **Real OS Sockets**: Uses genuine `socket` / `websockets` / `urllib` transport.
2. **Buffer Ingestion**: Feeds actual binary/audio/text streams rather than mocked function calls.
3. **Clean Teardown**: Leaves zero lingering background threads, sockets in `CLOSE_WAIT`, or unhandled event-loop exceptions.
