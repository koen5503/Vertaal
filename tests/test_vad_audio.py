"""Unit and regression tests for VAD chunk sizing and audio processing utilities."""

import math
import struct
import unittest
from tests.reference_logic import (
    calculate_rms,
    apply_gain,
    resample_for_stt,
    validate_webrtc_frame,
    CHUNK_SIZE,
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    CHANNELS,
)


class TestAudioUtilities(unittest.TestCase):
    """Test suite for RMS calculation, gain application, and STT resampling."""

    def test_calculate_rms_silence(self):
        """Pure zero bytes should yield exactly 0 RMS."""
        silence = b'\x00' * 960
        rms = calculate_rms(silence)
        self.assertEqual(rms, 0)

    def test_calculate_rms_empty(self):
        """Empty buffer should return 0 without division by zero."""
        self.assertEqual(calculate_rms(b''), 0)

    def test_calculate_rms_constant_dc(self):
        """Constant DC level audio should have RMS equal to the DC level."""
        amplitude = 8000
        data = struct.pack("<480h", *([amplitude] * 480))
        rms = calculate_rms(data)
        self.assertEqual(rms, amplitude)

    def test_calculate_rms_sine_wave(self):
        """Sine wave RMS should match theoretical amplitude / sqrt(2)."""
        amplitude = 10000
        freq = 440.0
        num_samples = 1600  # 100ms at 16kHz
        samples = [
            int(amplitude * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE)))
            for i in range(num_samples)
        ]
        data = struct.pack(f"<{num_samples}h", *samples)
        rms = calculate_rms(data)
        expected_rms = amplitude / math.sqrt(2)
        # Verify within 2% of theoretical RMS
        self.assertAlmostEqual(rms, expected_rms, delta=expected_rms * 0.02)

    def test_apply_gain_unity(self):
        """Gain factor of 1.0 should produce identical audio samples."""
        samples = [0, 500, -500, 15000, -15000, 30000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        gained = apply_gain(data, 1.0)
        self.assertEqual(gained, data)

    def test_apply_gain_zero(self):
        """Gain factor of 0.0 should yield all zeroes."""
        samples = [100, -200, 3000, -4000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        gained = apply_gain(data, 0.0)
        expected = b'\x00' * len(data)
        self.assertEqual(gained, expected)

    def test_apply_gain_attenuation_and_amplification(self):
        """Gain multiplication should scale amplitude proportionally."""
        samples = [1000, -2000, 4000]
        data = struct.pack("<3h", *samples)

        # 0.5x
        half_gain = apply_gain(data, 0.5)
        self.assertEqual(struct.unpack("<3h", half_gain), (500, -1000, 2000))

        # 2.0x
        double_gain = apply_gain(data, 2.0)
        self.assertEqual(struct.unpack("<3h", double_gain), (2000, -4000, 8000))

    def test_apply_gain_clipping(self):
        """Values exceeding int16 bounds [-32768, 32767] must clip cleanly."""
        samples = [20000, -20000, 30000, -32000]
        data = struct.pack("<4h", *samples)
        gained = apply_gain(data, 2.0)
        unpacked = struct.unpack("<4h", gained)
        self.assertEqual(unpacked, (32767, -32768, 32767, -32768))


class TestWebRTCVADChunkSizing(unittest.TestCase):
    """Verify that chunks for WebRTC VAD conform strictly to 10/20/30ms @ 16kHz."""

    def test_constants_match_spec(self):
        """Audio constants must match WebRTC VAD 30ms requirement at 16kHz mono."""
        self.assertEqual(SAMPLE_RATE, 16000)
        self.assertEqual(CHANNELS, 1)
        self.assertEqual(FRAME_DURATION_MS, 30)
        self.assertEqual(CHUNK_SIZE, 480)  # 480 samples
        # 16-bit mono = 2 bytes per sample -> 480 * 2 = 960 bytes
        expected_bytes = CHUNK_SIZE * 2
        self.assertEqual(expected_bytes, 960)

    def test_validate_webrtc_valid_frame_sizes(self):
        """10ms (160 samples), 20ms (320 samples), and 30ms (480 samples) must be valid."""
        for duration_ms, samples_count in [(10, 160), (20, 320), (30, 480)]:
            raw_bytes = b'\x00' * (samples_count * 2)
            self.assertTrue(
                validate_webrtc_frame(raw_bytes, sample_rate=16000),
                f"{duration_ms}ms frame of {samples_count} samples should be valid",
            )

    def test_validate_webrtc_invalid_frame_sizes(self):
        """Non-standard lengths must fail validation."""
        # 25ms = 400 samples (invalid for WebRTC VAD)
        self.assertFalse(validate_webrtc_frame(b'\x00' * 800, sample_rate=16000))
        # 30ms + 1 byte (odd byte count, invalid 16-bit)
        self.assertFalse(validate_webrtc_frame(b'\x00' * 961, sample_rate=16000))
        # 500 samples (31.25ms)
        self.assertFalse(validate_webrtc_frame(b'\x00' * 1000, sample_rate=16000))
        # Invalid sample rate
        self.assertFalse(validate_webrtc_frame(b'\x00' * 960, sample_rate=44100))


class TestResamplingConversions(unittest.TestCase):
    """Test resampling from various sample rates, channel layouts, and bit depths to 16kHz mono 16-bit."""

    def test_passthrough_16k_mono_16bit(self):
        """16kHz mono 16-bit input should pass through with identical sample count."""
        samples = [(50 * i) - 12000 for i in range(480)]
        data = struct.pack("<480h", *samples)
        result = resample_for_stt(data, src_rate=16000, src_channels=1, src_width=2)
        self.assertEqual(len(result), len(data))
        self.assertEqual(result, data)

    def test_resample_48k_stereo_to_16k_mono(self):
        """48kHz stereo 16-bit should downmix channels and decimate 3:1 to 16kHz mono."""
        num_input_frames = 4800  # 100ms at 48kHz
        # Left channel = 1000, Right channel = 3000 -> Expected mono average = 2000
        interleaved = []
        for _ in range(num_input_frames):
            interleaved.extend([1000, 3000])

        src_data = struct.pack(f"<{len(interleaved)}h", *interleaved)
        self.assertEqual(len(src_data), num_input_frames * 2 * 2)  # 19,200 bytes

        resampled = resample_for_stt(src_data, src_rate=48000, src_channels=2, src_width=2)

        # 100ms at 16kHz = 1600 samples = 3200 bytes
        expected_output_samples = int(num_input_frames * (16000 / 48000))
        self.assertEqual(len(resampled), expected_output_samples * 2)

        output_samples = struct.unpack(f"<{expected_output_samples}h", resampled)
        # Every sample should be 2000 (average of 1000 and 3000)
        for s in output_samples:
            self.assertEqual(s, 2000)

    def test_resample_44k1_stereo_to_16k_mono(self):
        """44.1kHz stereo 16-bit should convert to 16kHz mono without error."""
        num_input_frames = 44100  # 1.0 second
        # Interleaved stereo samples
        interleaved = []
        for i in range(num_input_frames):
            interleaved.extend([5000, 5000])

        src_data = struct.pack(f"<{len(interleaved)}h", *interleaved)
        resampled = resample_for_stt(src_data, src_rate=44100, src_channels=2, src_width=2)

        # 1.0s at 16kHz = 16000 samples = 32000 bytes
        expected_samples = int(num_input_frames / (44100 / 16000))
        self.assertEqual(len(resampled), expected_samples * 2)

        output_samples = struct.unpack(f"<{expected_samples}h", resampled)
        # Should preserve amplitude 5000
        for s in output_samples:
            self.assertEqual(s, 5000)

    def test_resample_8bit_unsigned_to_16bit(self):
        """8-bit unsigned PCM (where 128 is zero) should scale to signed 16-bit."""
        # 128 (center/zero), 255 (max positive), 0 (max negative)
        raw_8bit = bytes([128, 255, 0])
        resampled = resample_for_stt(raw_8bit, src_rate=16000, src_channels=1, src_width=1)
        unpacked = struct.unpack("<3h", resampled)
        self.assertEqual(unpacked[0], 0)            # (128 - 128) << 8 = 0
        self.assertEqual(unpacked[1], 127 << 8)     # (255 - 128) << 8 = 32512
        self.assertEqual(unpacked[2], -128 << 8)    # (0 - 128) << 8 = -32768

    def test_resample_32bit_signed_to_16bit(self):
        """32-bit signed PCM should shift down by 16 bits to 16-bit signed."""
        raw_32bit = struct.pack("<3i", 0, 10000 << 16, -20000 << 16)
        resampled = resample_for_stt(raw_32bit, src_rate=16000, src_channels=1, src_width=4)
        unpacked = struct.unpack("<3h", resampled)
        self.assertEqual(unpacked, (0, 10000, -20000))


if __name__ == "__main__":
    unittest.main()
