"""Run a canonical Agora Server SDK initialization smoke test.

This script is intentionally standalone so it can run in a Modal Function,
Modal VM Sandbox, or a regular Linux VM without importing the production worker.
The caller must provide AGORA_APP_ID through the environment.
"""

from __future__ import annotations

import json
import os
import platform
import site
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = "/tmp/agora-smoke/agorasdk.log"
DEFAULT_RESULT_PATH = "/tmp/agora-smoke/result.json"


def _log(message: str) -> None:
    print(f"[agora-smoke] {message}", flush=True)
    sys.stdout.flush()


def _find_agora_sdk_dir() -> Path:
    """Resolve the installed Agora SDK directory for the current Python."""
    candidates: list[Path] = []
    for site_dir in site.getsitepackages():
        candidates.append(Path(site_dir) / "agora" / "agora_sdk")
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "agora" / "agora_sdk")

    for candidate in candidates:
        if (candidate / "libagora_rtc_sdk.so").exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"Agora native SDK directory not found; searched: {searched}")


def _configure_native_environment() -> Path:
    sdk_dir = _find_agora_sdk_dir()
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(sdk_dir)]
    if existing:
        paths.append(existing)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    _log(f"sdk_dir={sdk_dir}")
    _log(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    return sdk_dir


def _write_result(result_path: Path, result: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_agora_smoke_test() -> dict[str, Any]:
    """Initialize and release Agora once using the documented configuration."""
    started = time.time()
    app_id = os.environ.get("AGORA_APP_ID", "").strip()
    if not app_id:
        raise RuntimeError(
            "AGORA_APP_ID is missing. Add AGORA_APP_ID to the Modal Secret "
            "or environment before running this test."
        )

    log_path = Path(os.environ.get("AGORA_LOG_PATH", DEFAULT_LOG_PATH))
    result_path = Path(os.environ.get("AGORA_RESULT_PATH", DEFAULT_RESULT_PATH))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _log(
        f"runtime={platform.platform()} python={platform.python_version()} "
        f"pid={os.getpid()} app_id_present=True app_id_length={len(app_id)}"
    )
    _log(f"log_path={log_path}")
    sdk_dir = _configure_native_environment()

    _log("importing agora.rtc")
    import agora.rtc  # noqa: F401

    _log("importing AgoraService symbols")
    from agora.rtc.agora_base import AudioScenarioType
    from agora.rtc.agora_service import AgoraService, AgoraServiceConfig

    _log("constructing documented AgoraServiceConfig")
    config = AgoraServiceConfig()
    config.appid = app_id
    config.audio_scenario = AudioScenarioType.AUDIO_SCENARIO_CHORUS
    config.enable_audio_processor = 1
    config.enable_audio_device = 0
    config.enable_video = 1
    config.use_string_uid = 0
    config.log_path = str(log_path)

    _log(
        "config prepared: audio_scenario=CHORUS "
        "enable_audio_processor=1 enable_audio_device=0 enable_video=1 "
        f"log_path_is_file={log_path.name == 'agorasdk.log'}"
    )
    _log("constructing AgoraService")
    service = AgoraService()

    _log("calling service.initialize(config)")
    result = service.initialize(config)
    _log(f"service.initialize returned result={result}")

    if result not in (None, 0):
        raise RuntimeError(f"AgoraService.initialize returned non-zero result: {result}")

    _log("calling service.release()")
    service.release()
    _log("service.release completed")

    output = {
        "status": "success",
        "result": result,
        "sdk_dir": str(sdk_dir),
        "log_path": str(log_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_result(result_path, output)
    return output


def main() -> int:
    try:
        result = run_agora_smoke_test()
    except Exception as error:
        result = {
            "status": "python_error",
            "error": str(error),
        }
        result_path = Path(os.environ.get("AGORA_RESULT_PATH", DEFAULT_RESULT_PATH))
        _write_result(result_path, result)
        _log(f"ERROR: {error}")
        return 1

    _log(f"result={json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
