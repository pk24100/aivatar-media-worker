import pytest

from streaming.core.audio_bus import CanonicalAudioBus
from streaming.orchestration.guards import reject_parked_session_config
from streaming.transport.websocket_server import WebsocketIngestionServer, _create_input_adapter


def test_handler_rejects_parked_ingestion_and_agora():
    with pytest.raises(ValueError, match="Invalid telephonyProvider"):
        reject_parked_session_config({"telephonyProvider": "twilio"})
    with pytest.raises(ValueError, match="Invalid inputProvider"):
        reject_parked_session_config({"inputProvider": "retell"})
    with pytest.raises(ValueError, match="Invalid inputProvider"):
        reject_parked_session_config({"inputProvider": "retell_ai"})
    with pytest.raises(ValueError, match="Invalid inputProvider"):
        reject_parked_session_config({"inputProvider": "retellai"})
    with pytest.raises(ValueError, match="Unsupported egress type"):
        reject_parked_session_config({"egressType": "agora"})
    with pytest.raises(ValueError, match="Unsupported egress type"):
        reject_parked_session_config({"room": {"type": "agora"}})


def test_handler_accepts_websocket_deepgram_and_livekit():
    reject_parked_session_config(
        {
            "inputProvider": "deepgram",
            "egressType": "livekit",
        }
    )
    reject_parked_session_config({})


def test_websocket_server_refuses_telephony_and_retell_adapters():
    bus = CanonicalAudioBus()
    with pytest.raises(ValueError, match="Invalid telephonyProvider"):
        _create_input_adapter(
            bus,
            session_id="s1",
            input_provider=None,
            telephony_provider="twilio",
        )
    with pytest.raises(ValueError, match="Invalid inputProvider"):
        _create_input_adapter(
            bus,
            session_id="s1",
            input_provider="retell_ai",
            telephony_provider=None,
        )


def test_websocket_server_register_session_refuses_parked_config():
    server = WebsocketIngestionServer()
    with pytest.raises(ValueError, match="Invalid telephonyProvider"):
        server.register_session("s1", "token", telephony_provider="exotel")
    with pytest.raises(ValueError, match="Invalid inputProvider"):
        server.register_session("s1", "token", input_provider="retell")


@pytest.mark.parametrize(
    ("input_provider", "expected_profile"),
    [
        ("deepgram", "deepgram"),
        ("gemini", "gemini"),
        ("gnani", None),
        ("elevenlabs", None),
        ("openai", None),
        ("cartesia", None),
        ("sarvam", None),
        ("custom", None),
    ],
)
def test_d8_input_providers_only_instantiate_behavior_profiles(
    input_provider,
    expected_profile,
):
    adapter = _create_input_adapter(
        CanonicalAudioBus(),
        session_id=f"session-{input_provider}",
        input_provider=input_provider,
        telephony_provider=None,
    )

    profile = adapter.provider_profile
    assert (profile.name if profile is not None else None) == expected_profile
