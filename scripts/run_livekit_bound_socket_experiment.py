"""Drive the deployed bound-socket experiment across several cold containers.

The unit of variation is the container, not the iteration, so this runner
invokes the deployed class repeatedly and waits past ``scaledown_window``
between calls so each invocation lands on a fresh cold or snapshot-restored
container. All cohorts are aggregated into one verdict file.

    modal deploy scripts/modal_livekit_bound_socket_experiment.py
    python scripts/run_livekit_bound_socket_experiment.py --cold-starts 6

Point it at an exact endpoint captured from a failing LIVEKIT_RTC_DEBUG log with:

    python scripts/run_livekit_bound_socket_experiment.py \
        --target udp:ip-161-115-167-190.host.livekit.cloud:3478 \
        --target tcp:ip-161-115-167-190.host.livekit.cloud:443
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve().parent / "modal_livekit_bound_socket_experiment.py"
_SPEC = importlib.util.spec_from_file_location("modal_livekit_bound_socket_experiment", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)


def _parse_target(text: str) -> dict[str, Any]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"--target must be proto:host:port, got {text!r}")
    protocol, host, port_text = parts
    protocol = protocol.strip().lower()
    if protocol not in {"udp", "tcp"}:
        raise ValueError(f"--target protocol must be udp or tcp, got {protocol!r}")
    return {
        "host": host.strip(),
        "port": int(port_text),
        "protocol": protocol,
        "scheme": "turn" if protocol == "udp" else "turns",
        "source": "explicit",
    }


def _verdict(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    total_sockets = sum(cohort["summary"]["socket_total"] for cohort in cohorts)
    total_enetunreach = sum(cohort["summary"]["enetunreach_total"] for cohort in cohorts)
    bound = Counter()
    unbound = Counter()
    for cohort in cohorts:
        bound.update(cohort["summary"]["by_variant"].get("bound", {}))
        unbound.update(cohort["summary"]["by_variant"].get("unbound", {}))
    restores = sum(1 for cohort in cohorts if cohort["boot_delta_seconds"] > 5)
    bound_fail = bound.get("errno_101_enetunreach", 0)
    unbound_fail = unbound.get("errno_101_enetunreach", 0)
    if total_enetunreach == 0:
        conclusion = "NOT_REPRODUCED: zero ENETUNREACH; cannot confirm or refute the bound-socket hypothesis"
    elif bound_fail > 0 and unbound_fail == 0:
        conclusion = "HYPOTHESIS_SUPPORTED: only bound sockets hit ENETUNREACH"
    elif bound_fail == 0 and unbound_fail > 0:
        conclusion = "INVERTED: only unbound sockets hit ENETUNREACH"
    else:
        conclusion = "REPRODUCED_BUT_NOT_BIND_RELATED: both variants hit ENETUNREACH"
    return {
        "cohorts": len(cohorts),
        "verified_snapshot_restores": restores,
        "socket_total": total_sockets,
        "enetunreach_total": total_enetunreach,
        "bound": dict(bound),
        "unbound": dict(unbound),
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cold-starts", type=int, default=6, help="Number of separate container invocations")
    parser.add_argument("--cooldown-seconds", type=float, default=60.0, help="Wait between invocations so the container scales to zero")
    parser.add_argument("--target", action="append", default=[], help="Explicit proto:host:port target; repeatable. Skips JoinResponse harvest.")
    parser.add_argument("--bursts", type=int, default=8, help="ICE-like port-set bursts per container")
    parser.add_argument("--sockets-per-target", type=int, default=4)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.4)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--harvest-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if not 1 <= args.cold_starts <= 50:
        raise ValueError("--cold-starts must be between 1 and 50")
    if not 1 <= args.bursts <= 100:
        raise ValueError("--bursts must be between 1 and 100")
    if not 1 <= args.sockets_per_target <= 32:
        raise ValueError("--sockets-per-target must be between 1 and 32")
    explicit_targets = [_parse_target(item) for item in args.target]

    import modal

    cls = modal.Cls.from_name(experiment._APP_NAME, "SnapshotSocketExperiment")
    cohorts: list[dict[str, Any]] = []
    for attempt in range(args.cold_starts):
        if attempt > 0:
            print(f"[runner] waiting {args.cooldown_seconds:.0f}s for scale-to-zero before cohort {attempt + 1}", flush=True)
            time.sleep(args.cooldown_seconds)
        print(f"[runner] cohort {attempt + 1}/{args.cold_starts} starting", flush=True)
        try:
            result = cls().run.remote(
                explicit_targets,
                args.bursts,
                args.sockets_per_target,
                args.hold_seconds,
                args.timeout_seconds,
                args.interval_seconds,
                args.harvest_timeout,
            )
        except Exception as exc:
            print(f"[runner] cohort {attempt + 1} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        cohorts.append(result)
        summary = result["summary"]
        boot = "restore" if result["boot_delta_seconds"] > 5 else "fresh"
        print(
            f"[runner] cohort {attempt + 1} {boot} boot_delta={result['boot_delta_seconds']}s "
            f"targets={result['target_source']} sockets={summary['socket_total']} "
            f"enetunreach={summary['enetunreach_total']}",
            flush=True,
        )

    if not cohorts:
        print("[runner] no cohort succeeded", flush=True)
        return 1

    verdict = _verdict(cohorts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).resolve().parent / f"livekit_bound_socket_verdict_{timestamp}.json"
    output_path.write_text(json.dumps({"verdict": verdict, "cohorts": cohorts}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
