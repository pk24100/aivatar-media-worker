import asyncio
import logging
from typing import Optional

from livekit import rtc

logger = logging.getLogger(__name__)

class SipAudioSubscriber:
    """
    Handles audio ingestion specifically from SIP trunks connected via LiveKit.
    When a SIP participant dials into the room, LiveKit publishes an audio track.
    This class subscribes to that track and routes it to the streaming engine queue.
    """
    # Initialize SIP subscriber with room and target audio queue.
    def __init__(
        self,
        room: rtc.Room,
        audio_queue: asyncio.Queue,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        self.room = room
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self._ready = asyncio.Event()
        self._audio_task = None
        self._track_sid = None

    # Bind to incoming audio tracks from SIP participants.
    def bind(self):
        # Callback fired when a remote audio track is subscribed in the SIP room.
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
                
            # Log info to help debug SIP connections
            logger.info(f"Subscribed to audio track from participant: {participant.identity} (SIP: {getattr(participant, 'sip', False)})")
            
            if self._audio_task is not None:
                logger.warning("Already consuming an audio track. Ignoring new track.")
                return
                
            self._track_sid = publication.sid
            audio_stream = rtc.AudioStream(
                track,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            self._audio_task = asyncio.create_task(self._consume(audio_stream))
            self._ready.set()

        # Check if SIP track is already published
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track and track.kind == rtc.TrackKind.KIND_AUDIO:
                    on_track_subscribed(track, publication, participant)
                    return

    # Wait until a SIP audio track has been subscribed.
    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            await self._ready.wait()
            return True
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # Consume SIP audio frames and enqueue them for processing.
    async def _consume(self, audio_stream: rtc.AudioStream):
        import numpy as np
        logger.info("Starting SIP audio consumption stream")
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                data = np.frombuffer(frame.data, dtype=np.int16)
                if self.num_channels > 1:
                    data = data.reshape(-1, self.num_channels).mean(axis=1)
                
                # Convert to float32 and scale
                audio = data.astype(np.float32) / 32768.0
                
                # Ensure 2D shape (time, channels) for downstream consumers
                if len(audio.shape) == 1:
                    audio = audio[:, np.newaxis]
                    
                if self.audio_queue.full():
                    try:
                        _ = self.audio_queue.get_nowait()
                        logger.warning("Audio queue full, dropping oldest frame")
                    except asyncio.QueueEmpty:
                        pass
                
                await self.audio_queue.put(audio)
        except Exception as e:
            logger.error(f"Error consuming SIP audio stream: {e}")
        finally:
            logger.info("SIP audio stream ended")
            await self.audio_queue.put(None)
