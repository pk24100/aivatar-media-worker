"""Factory for transport-neutral media egress adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from streaming.core.session import MediaEgressAdapter
from streaming.core.upscale import get_output_size

from .daily_egress import DailyEgressAdapter
from .livekit_egress import LiveKitEgressAdapter


class EgressAdapterFactory:
    """Create an egress adapter from an explicit type or room URL."""

    SUPPORTED_TYPES = frozenset({"livekit", "daily"})

    @staticmethod
    def create(
        egress_type: str | None,
        room_config: Mapping[str, Any],
    ) -> MediaEgressAdapter:
        config = dict(room_config)
        selected_type = (egress_type or config.get("type") or "").strip().lower()
        if not selected_type:
            selected_type = EgressAdapterFactory._detect_from_url(str(config.get("url", "")))
        if selected_type not in EgressAdapterFactory.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported egress type: {selected_type}")

        default_width, default_height = get_output_size()
        common = {
            "fps": int(config.get("fps", 25)),
            "sample_rate": int(config.get("sample_rate", 16_000)),
            "channels": int(config.get("channels", 1)),
            "session_id": str(config.get("session_id", "")),
        }
        if selected_type == "livekit":
            return LiveKitEgressAdapter(
                room=config.get("room"),
                room_url=config.get("room_url") or config.get("url"),
                room_token=config.get("room_token") or config.get("token"),
                **common,
            )
        return DailyEgressAdapter(
            room_url=config.get("room_url") or config.get("url"),
            room_token=config.get("room_token") or config.get("token"),
            width=int(config.get("width", default_width)),
            height=int(config.get("height", default_height)),
            **common,
        )

    @staticmethod
    def _detect_from_url(url: str) -> str:
        normalized = url.lower()
        if "livekit" in normalized or ".lk." in normalized:
            return "livekit"
        if "daily.co" in normalized or ".daily." in normalized:
            return "daily"
        raise ValueError(f"Cannot auto-detect egress type from URL: {url}")


__all__ = ["EgressAdapterFactory"]
