"""Utility functions and environment configuration helpers."""

import os
import sys
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

# Mock pkg_resources for webrtcvad if missing
if 'pkg_resources' not in sys.modules:
    class MockDistribution:
        version = '2.0.10'

    class MockPkgResources:
        @staticmethod
        def get_distribution(name):
            return MockDistribution()

    sys.modules['pkg_resources'] = MockPkgResources()


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_environment() -> None:
    """Load .env file from PyInstaller bundle or local working directory."""
    env_path = resource_path(".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def parse_bool(val: Any) -> bool:
    """Parse string or boolean to boolean value."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def resolve_pipeline_mode(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve pipeline mode from environment dictionary."""
    if env is None:
        env = os.environ
    return env.get("PIPELINE_MODE", "local").lower()


def get_runpod_gpu_fallbacks(preferred_gpu_or_env: Any = None) -> List[str]:
    """Return deduplicated list of candidate GPUs with preferred GPU first.
    
    Accepts an explicit GPU string, an environment dictionary, or defaults to os.environ.
    """
    if isinstance(preferred_gpu_or_env, dict):
        preferred_gpu = preferred_gpu_or_env.get("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 3090")
    elif isinstance(preferred_gpu_or_env, str):
        preferred_gpu = preferred_gpu_or_env
    else:
        preferred_gpu = os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 3090")

    candidate_gpus = [
        preferred_gpu,
        "NVIDIA GeForce RTX 3090",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA L4",
        "NVIDIA RTX 6000 Ada Generation",
    ]
    return list(dict.fromkeys(candidate_gpus))
