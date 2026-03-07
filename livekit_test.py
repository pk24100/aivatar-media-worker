import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()  # Load environment variables from .env file

def generate_livekit_token(room_name: str, participant_identity: str, api_key: str, api_secret: str) -> str:
    missing_values = {
        name: value
        for name, value in {
            "room_name": room_name,
            "participant_identity": participant_identity,
            "LIVEKIT_API_KEY": api_key,
            "LIVEKIT_API_SECRET": api_secret,
        }.items()
        if not value
    }
    if missing_values:
        missing = ", ".join(missing_values)
        raise ValueError(f"Missing required values for token generation: {missing}")

    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_identity)
    token.with_name("Test User")  # Optional

    video_grants = api.VideoGrants(
        room=room_name,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
    )

    token.with_grants(video_grants)

    return token.to_jwt()

# Example usage
room_name = os.getenv("LIVEKIT_ROOM", "test_room")
participant_identity = os.getenv("LIVEKIT_PARTICIPANT", "test_user")
api_key = os.getenv("LIVEKIT_API_KEY")
api_secret = os.getenv("LIVEKIT_API_SECRET")

token = generate_livekit_token(room_name, participant_identity, api_key, api_secret)
print(f"LiveKit Token: {token}")