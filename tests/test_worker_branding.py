from pathlib import Path

from streaming.transport.livekit.audio_publisher import AudioPublisher
from streaming.transport.livekit.video_publisher import VideoPublisher


ROOT = Path(__file__).resolve().parents[1]


def test_livekit_publishers_default_to_facemode_track_names():
    room = object()

    assert VideoPublisher(room).track_name == "facemode-video"
    assert AudioPublisher(room).track_name == "facemode-audio"


def test_active_scripts_mint_facemode_worker_identities():
    script_paths = (
        ROOT / "scripts" / "modal_concurrent_test.py",
        ROOT / "modal_app_stress.py",
    )

    for script_path in script_paths:
        source = script_path.read_text(encoding="utf-8")
        assert "identity=f\"facemode-worker-" in source
        assert "identity=f\"aivatar-worker-" not in source


def test_active_websocket_script_uses_facemode_subprotocol():
    source = (ROOT / "scripts" / "modal_concurrent_test.py").read_text(
        encoding="utf-8"
    )

    assert 'subprotocols=[f"facemode.{ws_token}"]' in source
    assert 'subprotocols=[f"aivatar.{ws_token}"]' not in source
