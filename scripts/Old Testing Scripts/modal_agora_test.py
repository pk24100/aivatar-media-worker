"""Compare Agora initialization in a Modal Function and a Modal VM Sandbox.

Prerequisite:
    Set the real Agora App ID in the local PowerShell environment:

        $env:AGORA_APP_ID = '<real Agora App ID>'

Run with:

    modal run modal_agora_test.py

The local entrypoint writes ``agora_runtime_test_results.log``. A SIGSEGV in
one remote environment is recorded as a crashed test so the comparison can
continue to the other environment.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import modal

AGORA_RESULT_PATH = "/tmp/agora-smoke/result.json"

smoke_image = (
    modal.Image.from_registry("python:3.12-slim")
    .apt_install(
        "ca-certificates",
        "libasound2",
        "libpulse0",
        "libx11-6",
        "libxcomposite1",
        "libxext6",
        "libxfixes3",
        "libxdamage1",
        "libglib2.0-0",
        "libnss3",
        "libgbm1",
        "libdrm2",
    )
    .pip_install("agora-python-server-sdk==2.4.9")
    .add_local_file("agora_smoke_test.py", "/root/agora_smoke_test.py", copy=True)
)

app = modal.App("aivatar-agora-runtime-test")


@app.function(
    image=smoke_image,
    timeout=180,
    retries=0,
)
def test_modal_function(app_id: str) -> dict[str, Any]:
    """Run the canonical test in a regular Modal Function runtime."""
    import sys

    os.environ["AGORA_APP_ID"] = app_id
    os.environ["AGORA_RESULT_PATH"] = AGORA_RESULT_PATH
    sys.path.insert(0, "/root")
    from agora_smoke_test import run_agora_smoke_test

    return run_agora_smoke_test()


def _run_vm_sandbox_test(app_id: str) -> dict[str, Any]:
    """Run the same image and script on Modal's full-kernel VM Sandbox."""
    sandbox = modal.Sandbox.create(
        app=app,
        image=smoke_image,
        timeout=300,
        experimental_options={"vm_runtime": True},
    )
    try:
        process = sandbox.exec(
            "python",
            "/root/agora_smoke_test.py",
            env={
                "AGORA_APP_ID": app_id,
                "AGORA_RESULT_PATH": AGORA_RESULT_PATH,
            },
            timeout=240,
        )
        process.wait()
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        return {
            "status": "completed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    finally:
        sandbox.terminate()


def _format_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


@app.local_entrypoint()
def main() -> None:
    output_file = Path("agora_runtime_test_results.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    app_id = os.environ.get("AGORA_APP_ID", "").strip()
    if not app_id:
        raise RuntimeError(
            "AGORA_APP_ID is missing from the local environment. "
            "Set it before running modal_agora_test.py."
        )

    report: list[str] = [
        f"=== Agora Runtime Comparison ({timestamp}) ===",
        "App ID source: local AGORA_APP_ID environment variable",
        f"App ID present: True; length: {len(app_id)}",
        "",
    ]

    print("# Running regular Modal Function test")
    try:
        function_result = test_modal_function.remote(app_id)
        report.extend(
            [
                "[modal_function] status=completed",
                _format_result(function_result),
            ]
        )
    except Exception as error:
        function_result = {"status": "crashed", "error": str(error)}
        report.extend(
            [
                "[modal_function] status=crashed",
                str(error),
            ]
        )
    print(_format_result(function_result))

    print("# Running full-kernel Modal VM Sandbox test")
    try:
        vm_result = _run_vm_sandbox_test(app_id)
        report.extend(
            [
                "",
                "[modal_vm_sandbox] status=completed",
                _format_result(vm_result),
            ]
        )
    except Exception as error:
        vm_result = {"status": "failed_to_run", "error": str(error)}
        report.extend(
            [
                "",
                "[modal_vm_sandbox] status=failed_to_run",
                str(error),
            ]
        )
    print(_format_result(vm_result))

    report.extend(
        [
            "",
            "=== Interpretation ===",
            "Function fails + VM succeeds: likely gVisor/runtime compatibility.",
            "Function fails + VM fails: likely Agora/config/system dependency issue.",
            "Function succeeds + VM succeeds: previous worker configuration was likely the cause.",
        ]
    )
    output_file.write_text("\n".join(report), encoding="utf-8")
    print(f"Results saved to: {output_file.resolve()}")
