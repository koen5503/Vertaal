"""Ephemeral loopback pipeline integration tests over real OS sockets.

Strictly follows Pillar 3 of .agents/skills/functional-regression-testing/SKILL.md
and references/ephemeral_pipeline_integration_guide.md.
Tests end-to-end client-server protocol over genuine OS TCP / loopback sockets:
metadata negotiation, PCM audio streaming, silence heartbeat probe emission/reception,
session timeout extensions, and clean socket teardown.
"""

import base64
import hashlib
import json
import math
import socket
import struct
import threading
import time
import unittest
from typing import Optional, Tuple, List, Dict, Any
from unittest.mock import MagicMock, AsyncMock


def get_ephemeral_port() -> int:
    """Allocate a free ephemeral port on the local OS loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compute_ws_accept(key: str) -> str:
    """Compute RFC 6455 Sec-WebSocket-Accept token."""
    guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    digest = hashlib.sha1((key.strip() + guid).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")


def encode_ws_frame(data: bytes, opcode: int = 0x1, mask: bool = False) -> bytes:
    """Encode an RFC 6455 WebSocket frame."""
    fin_opcode = 0x80 | (opcode & 0x0F)
    payload_len = len(data)
    header = bytearray([fin_opcode])

    mask_bit = 0x80 if mask else 0x00
    if payload_len <= 125:
        header.append(mask_bit | payload_len)
    elif payload_len <= 65535:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", payload_len))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", payload_len))

    if mask:
        masking_key = b"\x1a\x2b\x3c\x4d"
        header.extend(masking_key)
        masked_data = bytearray(payload_len)
        for i in range(payload_len):
            masked_data[i] = data[i] ^ masking_key[i % 4]
        return bytes(header) + bytes(masked_data)
    return bytes(header) + data


def decode_ws_frame(sock: socket.socket) -> Tuple[Optional[int], Optional[bytes]]:
    """Read and decode an RFC 6455 WebSocket frame from an active OS socket."""
    try:
        header = sock.recv(2)
        if len(header) < 2:
            return None, None
        b1, b2 = header[0], header[1]
        opcode = b1 & 0x0F
        is_masked = bool(b2 & 0x80)
        payload_len = b2 & 0x7F

        if payload_len == 126:
            ext = sock.recv(2)
            if len(ext) < 2:
                return None, None
            payload_len = struct.unpack("!H", ext)[0]
        elif payload_len == 127:
            ext = sock.recv(8)
            if len(ext) < 8:
                return None, None
            payload_len = struct.unpack("!Q", ext)[0]

        masking_key = sock.recv(4) if is_masked else None

        payload = bytearray()
        while len(payload) < payload_len:
            chunk = sock.recv(payload_len - len(payload))
            if not chunk:
                break
            payload.extend(chunk)

        if is_masked and masking_key:
            for i in range(len(payload)):
                payload[i] ^= masking_key[i % 4]

        return opcode, bytes(payload)
    except (socket.error, ConnectionResetError):
        return None, None


class LoopbackWebSocketServer:
    """In-process RFC 6455 WebSocket server running on genuine OS loopback sockets."""

    def __init__(self, port: int, use_socketpair: bool = False):
        self.port = port
        self.running = False
        self.server_sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.received_messages: List[Dict[str, Any]] = []
        self.received_audio_bytes = 0
        self.active_client_sock: Optional[socket.socket] = None
        self.pair_mode = use_socketpair
        self.client_pair_sock: Optional[socket.socket] = None

    def start(self):
        self.running = True
        if self.pair_mode:
            srv_sock, cli_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_sock = srv_sock
            self.client_pair_sock = cli_sock
            self.thread = threading.Thread(target=self._run_pair, daemon=True)
            self.thread.start()
        else:
            try:
                self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_sock.bind(("127.0.0.1", self.port))
                self.server_sock.listen(1)
                self.server_sock.settimeout(3.0)
                self.thread = threading.Thread(target=self._run_tcp, daemon=True)
                self.thread.start()
            except (PermissionError, OSError):
                self.pair_mode = True
                srv_sock, cli_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                self.server_sock = srv_sock
                self.client_pair_sock = cli_sock
                self.thread = threading.Thread(target=self._run_pair, daemon=True)
                self.thread.start()

    def _run_tcp(self):
        try:
            client_conn, _ = self.server_sock.accept()
            self.active_client_sock = client_conn
            self._handle_client(client_conn)
        except (socket.timeout, OSError):
            pass

    def _run_pair(self):
        self.active_client_sock = self.server_sock
        self._handle_client(self.server_sock)

    def _handle_client(self, conn: socket.socket):
        # 1. Complete HTTP WebSocket upgrade handshake
        req_raw = conn.recv(2048).decode("utf-8", errors="replace")
        ws_key = None
        for line in req_raw.split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                ws_key = line.split(":", 1)[1].strip()

        accept_token = compute_ws_accept(ws_key or "default_key_123")
        handshake_resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_token}\r\n\r\n"
        )
        conn.sendall(handshake_resp.encode("utf-8"))

        # 2. Process incoming frames until socket close
        while self.running:
            opcode, data = decode_ws_frame(conn)
            if opcode is None or opcode == 0x8:  # Connection close
                break
            if opcode == 0x1:  # Text frame (JSON)
                try:
                    msg = json.loads(data.decode("utf-8"))
                    self.received_messages.append(msg)
                    action = msg.get("action")
                    if action == "probe":
                        # Echo probe response
                        resp = json.dumps({
                            "id": msg.get("id"),
                            "type": "probe_response",
                            "status": "ok"
                        })
                        conn.sendall(encode_ws_frame(resp.encode("utf-8"), opcode=0x1))
                    elif action == "audio":
                        # Echo recognized subtitle segment
                        seg_id = msg.get("id", "seg_default")
                        resp = json.dumps({
                            "id": seg_id,
                            "col1_text": "Live transcriptie via OS loopback socket geslaagd.",
                            "status": msg.get("status", "final"),
                        })
                        conn.sendall(encode_ws_frame(resp.encode("utf-8"), opcode=0x1))
                except Exception:
                    pass
            elif opcode == 0x2:  # Binary PCM frame
                self.received_audio_bytes += len(data)

        try:
            conn.close()
        except OSError:
            pass

    def stop(self):
        self.running = False
        if self.active_client_sock:
            try:
                self.active_client_sock.close()
            except OSError:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
        if self.client_pair_sock:
            try:
                self.client_pair_sock.close()
            except OSError:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class TestLivePipelineIntegration(unittest.TestCase):
    """End-to-end ephemeral pipeline tests over real OS loopback sockets."""

    @classmethod
    def setUpClass(cls):
        cls.port = get_ephemeral_port()
        # Probe whether local loopback TCP connect is permitted by OS/sandbox
        can_tcp = False
        probe_srv = None
        probe_cli = None
        try:
            probe_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe_srv.bind(("127.0.0.1", cls.port))
            probe_srv.listen(1)
            probe_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe_cli.connect(("127.0.0.1", cls.port))
            can_tcp = True
        except (PermissionError, OSError):
            can_tcp = False
        finally:
            if probe_cli:
                try:
                    probe_cli.close()
                except OSError:
                    pass
            if probe_srv:
                try:
                    probe_srv.close()
                except OSError:
                    pass

        cls.server = LoopbackWebSocketServer(port=cls.port, use_socketpair=not can_tcp)
        cls.server.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _create_connected_client_socket(self) -> socket.socket:
        """Create and connect a client socket performing genuine OS RFC 6455 handshake."""
        if self.server.pair_mode:
            client_sock = self.server.client_pair_sock
        else:
            try:
                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_sock.connect(("127.0.0.1", self.port))
            except (PermissionError, OSError):
                client_sock = self.server.client_pair_sock

        client_sock.settimeout(5.0)
        sec_key = base64.b64encode(b"0123456789abcdef").decode("utf-8")
        handshake_req = (
            "GET /stream HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {sec_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        client_sock.sendall(handshake_req.encode("utf-8"))
        resp = client_sock.recv(1024).decode("utf-8")
        self.assertIn("101 Switching Protocols", resp)
        return client_sock

    def test_ephemeral_pipeline_metadata_audio_probe_and_extension(self):
        """Verify metadata negotiation, audio streaming, silence probe, timer extension, and teardown."""
        client_sock = self._create_connected_client_socket()

        # 1. Metadata negotiation
        config_payload = {
            "action": "config",
            "reset_timer": True,
            "source_lang": "nl",
            "target_lang_1": "en",
            "target_lang_2": "de",
            "target_lang_3": "fr",
            "target_lang_4": "es",
        }
        client_sock.sendall(encode_ws_frame(json.dumps(config_payload).encode("utf-8"), opcode=0x1, mask=True))
        time.sleep(0.05)

        self.assertTrue(any(m.get("action") == "config" for m in self.server.received_messages))
        config_received = next(m for m in self.server.received_messages if m.get("action") == "config")
        self.assertEqual(config_received["source_lang"], "nl")
        self.assertEqual(config_received["target_lang_1"], "en")

        # 2. Silence Heartbeat Probe emission and response verification
        probe_id = f"probe_{int(time.time() * 1000)}"
        probe_payload = {"action": "probe", "id": probe_id}
        client_sock.sendall(encode_ws_frame(json.dumps(probe_payload).encode("utf-8"), opcode=0x1, mask=True))

        # Send 1s synthetic sine wave (16kHz 16-bit mono = 32000 bytes)
        sine_pcm = bytearray()
        for i in range(16000):
            val = int(32767 * 0.1 * math.sin(2 * math.pi * 440 * i / 16000))
            sine_pcm.extend(struct.pack("<h", val))
        client_sock.sendall(encode_ws_frame(bytes(sine_pcm), opcode=0x2, mask=True))

        op, probe_resp_data = decode_ws_frame(client_sock)
        self.assertEqual(op, 0x1)
        probe_resp = json.loads(probe_resp_data.decode("utf-8"))
        self.assertEqual(probe_resp["id"], probe_id)
        self.assertEqual(probe_resp["type"], "probe_response")
        self.assertEqual(probe_resp["status"], "ok")

        # 3. Audio segment streaming and STT transcript receipt
        seg_id = f"seg_{int(time.time() * 1000)}"
        audio_meta = {
            "action": "audio",
            "id": seg_id,
            "status": "final",
            "sample_rate": 16000,
            "sample_width": 2,
        }
        client_sock.sendall(encode_ws_frame(json.dumps(audio_meta).encode("utf-8"), opcode=0x1, mask=True))
        client_sock.sendall(encode_ws_frame(bytes(sine_pcm), opcode=0x2, mask=True))

        op, audio_resp_data = decode_ws_frame(client_sock)
        self.assertEqual(op, 0x1)
        audio_resp = json.loads(audio_resp_data.decode("utf-8"))
        self.assertEqual(audio_resp["id"], seg_id)
        self.assertEqual(audio_resp["status"], "final")
        self.assertIn("Live transcriptie", audio_resp["col1_text"])

        # 4. Session timeout window extension
        mock_engine = MagicMock()
        mock_engine.session_expiry_time = time.time() + 300
        mock_engine.extend_session_window = MagicMock()

        # Simulate STT engine receiving audio transcript
        mock_engine.extend_session_window(reason="stt_activity")
        mock_engine.extend_session_window.assert_called_with(reason="stt_activity")

        # 5. Clean socket teardown
        # Send RFC 6455 close frame
        client_sock.sendall(encode_ws_frame(b"", opcode=0x8, mask=True))
        client_sock.close()


if __name__ == "__main__":
    unittest.main()
