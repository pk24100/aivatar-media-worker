"""Refuse parked ingestion and egress before any adapter is constructed."""

from __future__ import annotations

from typing import Any, Mapping

PARKED_PROVIDER_PROFILES = frozenset({"retell", "retell_ai", "retellai"})


def normalize_optional_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def reject_parked_session_config(event: Mapping[str, Any] | None) -> None:
    event = event or {}
    room = event.get("room") if isinstance(event.get("room"), dict) else {}
    if event.get("telephonyProvider"):
        raise ValueError("Invalid telephonyProvider")
    profile_name = normalize_optional_name(event.get("inputProvider"))
    if profile_name in PARKED_PROVIDER_PROFILES:
        raise ValueError("Invalid inputProvider")
    egress_type = str(event.get("egressType") or room.get("type") or "").strip().lower()
    if egress_type == "agora":
        raise ValueError("Unsupported egress type")
