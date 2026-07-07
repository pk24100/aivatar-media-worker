"""
Orchestrator: run both Modal diagnostic scripts and save results locally.

Usage:
    cd aivatar-media-worker
    python scripts/modal_run_both_diags.py

This will:
1. Run modal_diag_function.py  (via modal run) and save JSON
2. Deploy modal_diag_webserver.py, curl the endpoint, save JSON, then stop it

Results are saved to:
  scripts/modal_diag_function_*.json   — structured JSON (includes container_logs)
  scripts/modal_diag_function_*.log    — raw Modal client + container stdout/stderr
  scripts/modal_diag_webserver_*.json  — structured JSON (includes container_logs)
  scripts/modal_diag_webserver_*.log   — raw Modal client + container stdout/stderr
"""

import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time


def run_command(cmd, cwd, timeout=600):
    """Run a shell command and return stdout/stderr."""
    print(f"\n[CMD] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0 and proc.stderr:
        print(f"[STDERR] {proc.stderr}")
    return proc


def main():
    worker_dir = pathlib.Path(__file__).parent.parent
    scripts_dir = worker_dir / "scripts"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -----------------------------------------------------------------------
    # 1. Function endpoint
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. RUNNING FUNCTION ENDPOINT")
    print("=" * 60)

    func_proc = run_command(
        ["python", "-m", "modal", "run", str(scripts_dir / "modal_diag_function.py")],
        cwd=str(worker_dir),
        timeout=300,
    )

    # Save raw subprocess output (Modal client logs + container stdout/stderr)
    func_log_file = scripts_dir / f"modal_diag_function_{ts}.log"
    raw_output = func_proc.stdout or ""
    if func_proc.stderr:
        raw_output += "\n\n--- STDERR ---\n" + func_proc.stderr
    func_log_file.write_text(raw_output, encoding="utf-8")
    print(f"[SAVED] Function raw logs -> {func_log_file}")

    # Extract JSON using markers printed by the local entrypoint
    func_result = "{}"
    marker_start = "===DIAG_JSON_START==="
    marker_end = "===DIAG_JSON_END==="
    if marker_start in raw_output and marker_end in raw_output:
        s = raw_output.index(marker_start) + len(marker_start)
        e = raw_output.index(marker_end)
        func_result = raw_output[s:e].strip()
    else:
        # Fallback: try to find last valid JSON object in output
        try:
            lines = [l for l in raw_output.splitlines() if l.strip()]
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    json.loads(line)
                    func_result = line
                    break
        except Exception:
            pass

    func_file = scripts_dir / f"modal_diag_function_{ts}.json"
    func_file.write_text(func_result, encoding="utf-8")
    print(f"[SAVED] Function diagnostics -> {func_file}")

    # -----------------------------------------------------------------------
    # 2. Web server endpoint
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. DEPLOYING + CURLING WEB_SERVER ENDPOINT")
    print("=" * 60)

    deploy_name = f"aivatar-diag-{ts}"
    deploy_proc = run_command(
        [
            "python", "-m", "modal", "deploy",
            str(scripts_dir / "modal_diag_webserver.py"),
            "--name", deploy_name,
        ],
        cwd=str(worker_dir),
        timeout=120,
    )

    # Save raw deploy output
    web_log_file = scripts_dir / f"modal_diag_webserver_{ts}.log"
    deploy_raw = deploy_proc.stdout or ""
    if deploy_proc.stderr:
        deploy_raw += "\n\n--- STDERR ---\n" + deploy_proc.stderr

    web_result = "{}"
    if deploy_proc.returncode != 0:
        print("[ERROR] Deploy failed.")
        web_result = json.dumps({"error": f"deploy failed: {deploy_proc.stderr}"})
    else:
        urls = re.findall(
            r'https://[a-zA-Z0-9\-._~:/?#@!$&\'()*+,;=%]+\.modal\.run',
            deploy_proc.stdout,
        )
        if urls:
            url = urls[0]
        else:
            url = f"https://pk24100--{deploy_name.replace('_', '-')}-web-server-diagnostics.modal.run"

        print(f"\n[URL] {url}")
        print("[INFO] Waiting 10s for container startup...")
        time.sleep(10)

        curl_proc = run_command(
            ["curl", "-s", "-m", "180", url],
            cwd=str(worker_dir),
            timeout=200,
        )
        if curl_proc.returncode == 0 and curl_proc.stdout:
            web_result = curl_proc.stdout
            print(f"[RESPONSE] {web_result[:500]}...")
            deploy_raw += "\n\n--- CURL RESPONSE ---\n" + web_result
        else:
            print(f"[ERROR] Curl failed: {curl_proc.stderr}")
            web_result = json.dumps({"error": f"curl failed: {curl_proc.stderr}"})
            deploy_raw += f"\n\n--- CURL ERROR ---\n{curl_proc.stderr}"

        # No automatic cleanup — user stops manually
        print(f"\n[INFO] Deployment '{deploy_name}' is still running.")
        print(f"  Stop it manually with:  python -m modal app stop {deploy_name}")

    web_log_file.write_text(deploy_raw, encoding="utf-8")
    print(f"[SAVED] WebServer raw logs -> {web_log_file}")

    web_file = scripts_dir / f"modal_diag_webserver_{ts}.json"
    web_file.write_text(web_result, encoding="utf-8")
    print(f"[SAVED] WebServer diagnostics -> {web_file}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print(f"  Function JSON:  {func_file}")
    print(f"  Function LOG:   {func_log_file}")
    print(f"  WebServer JSON: {web_file}")
    print(f"  WebServer LOG:  {web_log_file}")
    print("\nCompare the JSON files for structured results.")
    print("Check the LOG files for complete Modal + container logs.")
    print("The JSON files also contain a 'container_logs' field with all in-container logs.")


if __name__ == "__main__":
    main()
