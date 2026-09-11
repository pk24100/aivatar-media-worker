from unittest.mock import Mock

import pytest

from streaming.transport.adapters import factory
from streaming.transport.adapters.factory import EgressAdapterFactory


@pytest.mark.parametrize(
    ("egress_type", "url", "expected"),
    [
        ("livekit", "wss://rooms.example", "livekit"),
        ("daily", "https://example.daily.co/room", "daily"),
        (None, "wss://project.livekit.cloud", "livekit"),
        (None, "https://example.daily.co/room", "daily"),
    ],
)
def test_factory_selects_adapter_by_type_or_url(monkeypatch, egress_type, url, expected):
    classes = {
        "livekit": Mock(return_value=Mock(name="livekit-adapter")),
        "daily": Mock(return_value=Mock(name="daily-adapter")),
    }
    monkeypatch.setattr(factory, "LiveKitEgressAdapter", classes["livekit"])
    monkeypatch.setattr(factory, "DailyEgressAdapter", classes["daily"])

    adapter = EgressAdapterFactory.create(
        egress_type,
        {"url": url, "token": "token"},
    )

    assert adapter is classes[expected].return_value
    classes[expected].assert_called_once()


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported egress type"):
        EgressAdapterFactory.create("agora", {"url": "agora://channel-name", "token": "token"})


def test_factory_rejects_unrecognizable_url():
    with pytest.raises(ValueError, match="Cannot auto-detect egress type"):
        EgressAdapterFactory.create(None, {"url": "https://example.test", "token": "token"})
