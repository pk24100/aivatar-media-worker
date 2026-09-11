from __future__ import annotations

import errno
import importlib.util
import socket
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "modal_livekit_bound_socket_experiment.py"
_SPEC = importlib.util.spec_from_file_location("modal_livekit_bound_socket_experiment", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_livekit_bound_socket_experiment.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("run_livekit_bound_socket_experiment", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


def _record(**overrides):
    base = {
        "protocol": "udp",
        "port": 3478,
        "bound": True,
        "create_errno": None,
        "send_errno": None,
        "post_hold_errno": None,
        "response": None,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "turn:ip-161-115-167-190.host.livekit.cloud:3478?transport=udp",
            {"host": "ip-161-115-167-190.host.livekit.cloud", "port": 3478, "protocol": "udp", "scheme": "turn"},
        ),
        (
            "turns:otoronto1a.turn.livekit.cloud:443?transport=tcp",
            {"host": "otoronto1a.turn.livekit.cloud", "port": 443, "protocol": "tcp", "scheme": "turns"},
        ),
        (
            "stun:stun.example.com:3478",
            {"host": "stun.example.com", "port": 3478, "protocol": "udp", "scheme": "stun"},
        ),
    ],
)
def test_parse_ice_server_url_extracts_per_node_targets(url, expected):
    parsed = experiment._parse_ice_server_url(url)
    for key, value in expected.items():
        assert parsed[key] == value


@pytest.mark.parametrize("url", ["", "https://example.com", "not-a-url"])
def test_parse_ice_server_url_rejects_non_ice_urls(url):
    assert experiment._parse_ice_server_url(url) is None


def test_parse_ice_server_url_defaults_port_by_scheme():
    assert experiment._parse_ice_server_url("turns:example.com")["port"] == 5349
    assert experiment._parse_ice_server_url("turn:example.com")["port"] == 3478


def test_dedupe_targets_keeps_distinct_protocol_and_port():
    targets = [
        {"host": "a", "port": 3478, "protocol": "udp"},
        {"host": "a", "port": 3478, "protocol": "udp"},
        {"host": "a", "port": 3478, "protocol": "tcp"},
        {"host": "a", "port": 443, "protocol": "tcp"},
    ]
    assert len(experiment._dedupe_targets(targets)) == 3


def test_classify_prioritises_enetunreach_from_any_phase():
    assert experiment._classify(_record(create_errno=101)) == "errno_101_enetunreach"
    assert experiment._classify(_record(send_errno=errno.ENETUNREACH)) == "errno_101_enetunreach"
    assert experiment._classify(_record(post_hold_errno=101)) == "errno_101_enetunreach"


def test_classify_distinguishes_host_unreachable_and_other_errno():
    assert experiment._classify(_record(send_errno=113)) == "errno_113_ehostunreach"
    assert experiment._classify(_record(send_errno=111)) == "errno_111"


def test_classify_success_paths():
    assert experiment._classify(_record(response=True)) == "sent_response"
    assert experiment._classify(_record(response=False)) == "sent_no_response"
    assert experiment._classify(_record(protocol="tcp", port=443)) == "connected"


def test_summarize_separates_bound_from_unbound_and_counts_enetunreach():
    records = [
        _record(bound=True, send_errno=101),
        _record(bound=False, response=True),
        _record(bound=True, protocol="tcp", port=443),
    ]
    summary = experiment._summarize(records)
    assert summary["by_variant"]["bound"]["errno_101_enetunreach"] == 1
    assert summary["by_variant"]["unbound"]["sent_response"] == 1
    assert summary["by_target"]["tcp:443:bound"] == {"connected": 1}
    assert summary["enetunreach_total"] == 1
    assert summary["socket_total"] == 3


def test_derive_target_host_still_supports_fallback():
    assert experiment._derive_target_host("wss://example.livekit.cloud") == "example.turn.livekit.cloud"
    with pytest.raises(ValueError):
        experiment._derive_target_host("")


def test_run_burst_allocates_every_socket_before_sending(monkeypatch):
    """Burst semantics: all sockets must exist before the first send."""
    events: list[str] = []

    class FakeSocket:
        def __init__(self, *_args, **_kwargs):
            events.append("create")

        def settimeout(self, _value):
            pass

        def bind(self, _addr):
            pass

        def getsockname(self):
            return ("172.20.2.2", 40000)

        def sendto(self, _data, _addr):
            events.append("send")

        def recvfrom(self, _size):
            raise TimeoutError

        def close(self):
            pass

    monkeypatch.setattr(experiment.socket, "socket", FakeSocket)
    monkeypatch.setattr(experiment.time, "sleep", lambda _s: None)

    targets = [{"host": "h", "remote_ip": "203.0.113.5", "port": 3478, "protocol": "udp", "scheme": "turn", "source": "t"}]
    records = experiment._run_burst(targets, "172.20.2.2", 3, 0.0, 0.1)

    assert len(records) == 6  # 3 sockets x bound/unbound
    first_send = events.index("send")
    assert events[:first_send].count("create") == 6, "all sockets must be created before any send"


def test_runner_parses_explicit_targets():
    assert runner._parse_target("udp:ip-1-2-3-4.host.livekit.cloud:3478") == {
        "host": "ip-1-2-3-4.host.livekit.cloud",
        "port": 3478,
        "protocol": "udp",
        "scheme": "turn",
        "source": "explicit",
    }
    with pytest.raises(ValueError):
        runner._parse_target("udp:host")
    with pytest.raises(ValueError):
        runner._parse_target("sctp:host:1234")


@pytest.mark.parametrize(
    ("bound_fail", "unbound_fail", "expected"),
    [
        (0, 0, "NOT_REPRODUCED"),
        (5, 0, "HYPOTHESIS_SUPPORTED"),
        (0, 5, "INVERTED"),
        (5, 5, "REPRODUCED_BUT_NOT_BIND_RELATED"),
    ],
)
def test_runner_verdict_logic(bound_fail, unbound_fail, expected):
    cohort = {
        "boot_delta_seconds": 500.0,
        "summary": {
            "socket_total": 20,
            "enetunreach_total": bound_fail + unbound_fail,
            "by_variant": {
                "bound": {"errno_101_enetunreach": bound_fail} if bound_fail else {},
                "unbound": {"errno_101_enetunreach": unbound_fail} if unbound_fail else {},
            },
        },
    }
    verdict = runner._verdict([cohort])
    assert verdict["conclusion"].startswith(expected)
    assert verdict["verified_snapshot_restores"] == 1
