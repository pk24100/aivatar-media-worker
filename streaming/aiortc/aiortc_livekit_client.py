"""
Connects to a LiveKit room using aiortc for WebRTC transport.

Pure-Python LiveKit room client using aiortc for WebRTC transport.

This module implements LiveKit's signaling protocol (protobuf over WebSocket)
and uses aiortc's RTCPeerConnection for the media transport layer. It exists
because Modal's gVisor sandbox blocks the Rust-native livekit_ffi library
from establishing DTLS peer connections, while aiortc (pure Python) works.

The native livekit-rtc SDK path is kept for RunPod — see stream_processor.py
for the runtime backend selection.
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from typing import Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import websockets
from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from aiortc.sdp import candidate_from_sdp
from livekit.protocol import rtc as lk_rtc, models as lk_models

class LiveKitReconnectException(Exception):
    def __init__(self, url: str, reconnect: bool = False):
        self.url = url
        self.reconnect = reconnect
        super().__init__(f"Reconnect to {url} (resume={reconnect})")

_logger = logging.getLogger("aiortc_livekit_client")
# INFO keeps per-ICE-candidate DEBUG spam out of the logs while we investigate.
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    _logger.addHandler(_handler)


def _log(msg, *args):
    """Print + log so output always appears in Modal container logs."""
    formatted = msg % args if args else msg
    print(formatted, flush=True)
    _logger.info(formatted)


def _enum_name(enum_wrapper, value) -> str:
    """Best-effort protobuf enum value -> name (falls back to the raw value)."""
    try:
        return enum_wrapper.Name(value)
    except Exception:
        return str(value)


def _summarize_sdp(sdp: str) -> str:
    """Compact one-line SDP summary: media sections, datachannel + dtls setup."""
    media = []
    has_datachannel = False
    setup = None
    for line in sdp.splitlines():
        if line.startswith("m="):
            kind = line.split()[0][2:]
            media.append(kind)
            if kind == "application":
                has_datachannel = True
        elif line.startswith("a=setup:") and setup is None:
            setup = line.split(":", 1)[1]
    return f"m-lines={media} datachannel={has_datachannel} setup={setup} bytes={len(sdp)}"


class AiortcLiveKitClient:
    """
    Connects to a LiveKit room using aiortc for WebRTC.

    Uses separate publisher and subscriber PeerConnections matching LiveKit's
    signaling targets. The server sends a subscriber offer first; publisher
    tracks are negotiated on the publisher PeerConnection.

    Signaling uses protobuf (SignalRequest / SignalResponse) over WebSocket.
    """

    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._publisher_pc: Optional[RTCPeerConnection] = None
        self._subscriber_pc: Optional[RTCPeerConnection] = None
        self._connection_state = "disconnected"
        self._join_response: Optional[lk_rtc.JoinResponse] = None
        self._pending_candidates: Dict[int, List[dict]] = {
            lk_rtc.SignalTarget.PUBLISHER: [],
            lk_rtc.SignalTarget.SUBSCRIBER: [],
        }
        self._track_published_futures: Dict[str, asyncio.Future] = {}
        self._on_track_subscribed: Optional[Callable] = None
        self.reconnect_url: Optional[str] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._ice_config: Optional[RTCConfiguration] = None
        self._publisher_offer_sent_at: Optional[float] = None
        self._publisher_offer_id = 0
        self._original_url: Optional[str] = None

        # Events for flow control
        self._subscriber_answer_sent = asyncio.Event()
        self._publisher_answer_received = asyncio.Event()
        self._pc_ready = asyncio.Event()

    @property
    def connection_state(self) -> str:
        return self._connection_state

    def on_track_subscribed(self, callback: Callable):
        """Register a callback for when a remote track is subscribed.

        callback(track: MediaStreamTrack)
        """
        self._on_track_subscribed = callback

    # ------------------------------------------------------------------ #
    # Connect
    # ------------------------------------------------------------------ #

    async def connect(self, url: str, token: str, reconnect: bool = False) -> None:
        """Connect to a LiveKit room."""
        self._original_url = url
        ws_url = self._build_ws_url(url, token, reconnect=reconnect)
        _log("[aiortc] Connecting to LiveKit signaling: %s", ws_url[:100] + "...")

        try:
            # Do NOT send a WebSocket subprotocol header. The known-working
            # aiortc client (dguerizec/livekit-client-sdk-python) connects
            # without one, letting the server fall back to the `protocol=8`
            # query parameter. Sending `lk-protocol-13` overrides that and
            # forces the server into fast_publish mode, which our aiortc
            # publisher flow does not support.
            self._ws = await websockets.connect(
                ws_url,
                max_size=16 * 1024 * 1024,
                ping_interval=None,  # we handle pings ourselves
                open_timeout=30,  # Modal gVisor DNS + TCP can exceed default 10s
                close_timeout=10,
            )
        except TimeoutError:
            _log("[aiortc] WebSocket connect TIMEOUT after 30s — "
                 "DNS or network unreachable from this container")
            raise
        except OSError as exc:
            _log("[aiortc] WebSocket connect OS error: %s", exc)
            raise
        _log("[aiortc] WebSocket connected")

        # Wait for JoinResponse
        raw = await self._ws.recv()
        resp = lk_rtc.SignalResponse()
        resp.ParseFromString(raw if isinstance(raw, bytes) else raw.encode())

        if not resp.HasField("join"):
            raise ConnectionError(f"Expected JoinResponse, got: {resp.WhichOneof('message')}")

        self._join_response = resp.join
        _log(
            "[aiortc] JoinResponse: room=%s region=%s subscriber_primary=%s "
            "fast_publish=%s ice_servers=%d ping_interval=%d ping_timeout=%d",
            self._join_response.room.name,
            self._join_response.server_region,
            self._join_response.subscriber_primary,
            self._join_response.fast_publish,
            len(self._join_response.ice_servers),
            self._join_response.ping_interval,
            self._join_response.ping_timeout,
        )

        # Log ICE server URLs
        for i, srv in enumerate(self._join_response.ice_servers):
            _log("[aiortc]   ICE server %d: %s", i, list(srv.urls))

        # Build ICE configuration from server-provided ICE servers
        ice_servers = []
        for srv in self._join_response.ice_servers:
            ice_servers.append(
                RTCIceServer(
                    urls=list(srv.urls),
                    username=srv.username or None,
                    credential=srv.credential or None,
                )
            )
        if ice_servers:
            self._ice_config = RTCConfiguration(iceServers=ice_servers)

        self._publisher_pc = RTCPeerConnection(configuration=self._ice_config)
        # NOTE: aiortc creates a separate ICE transport for datachannels (SCTP)
        # which generates different ICE credentials (ufrag/pwd) in the SDP
        # application m-line vs the media m-lines. This violates BUNDLE
        # semantics and causes Pion to reject the publisher offer with
        # NegotiateFailed -> STATE_MISMATCH. The server already provides a
        # datachannel in the subscriber offer, so we rely on that for
        # bidirectional data. Publisher tracks are sent via the media m-lines.
        #
        # (Previously: createDataChannel("_reliable") and ("_lossy") were
        # created here, but they caused the BUNDLE ICE credential mismatch.)

        # NOTE: Do NOT pre-allocate recvonly transceivers here. The publisher
        # transport is unidirectional (client -> server); the LiveKit/Pion
        # server rejects a publisher offer that contains recvonly m-lines
        # (NegotiateFailed -> STATE_MISMATCH). The real LiveKit client SDK
        # (RTCEngine.ts addPublisherTransceiver) only adds sendonly
        # transceivers for the tracks being published. Tracks are added in
        # publish_tracks() with direction="sendonly".

        self._setup_pc(self._publisher_pc, lk_rtc.SignalTarget.PUBLISHER, "publisher")

        self._subscriber_pc = RTCPeerConnection(configuration=self._ice_config)
        self._setup_pc(self._subscriber_pc, lk_rtc.SignalTarget.SUBSCRIBER, "subscriber")
        self._pc_ready.set()
        _log("[aiortc] Publisher/subscriber PCs created")

        # Start listening for signaling messages
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        # Wait for the subscriber offer to arrive and be answered.
        # LiveKit requires the subscriber PC to be connected before publishing.
        _log("[aiortc] Waiting for subscriber offer from server...")
        try:
            await asyncio.wait_for(self._subscriber_answer_sent.wait(), timeout=15.0)
            _log("[aiortc] Subscriber offer answered — subscriber PC ready")
        except asyncio.TimeoutError:
            _logger.warning(
                "[aiortc] Subscriber offer not received/answered within 15s. "
                "subscriber_primary=%s — proceeding anyway",
                self._join_response.subscriber_primary,
            )

        # Wait for subscriber ICE to actually connect before publishing.
        # Sending a publisher offer while subscriber ICE is still "checking"
        # can cause the server to reject with STATE_MISMATCH.
        _log("[aiortc] Waiting for subscriber ICE to connect...")
        sub_ice_connected = asyncio.Event()

        @self._subscriber_pc.on("iceconnectionstatechange")
        def _wait_sub_ice():
            state = self._subscriber_pc.iceConnectionState
            if state in ("connected", "completed"):
                sub_ice_connected.set()

        if self._subscriber_pc.iceConnectionState in ("connected", "completed"):
            sub_ice_connected.set()

        try:
            await asyncio.wait_for(sub_ice_connected.wait(), timeout=10.0)
            _log("[aiortc] Subscriber ICE %s", self._subscriber_pc.iceConnectionState)
        except asyncio.TimeoutError:
            _log("[aiortc] WARNING: Subscriber ICE not connected within 10s, "
                 "proceeding anyway (state=%s)",
                 self._subscriber_pc.iceConnectionState)

        self._connection_state = "connected"
        _log("[aiortc] Connected to LiveKit room")

    # ------------------------------------------------------------------ #
    # Publish tracks
    # ------------------------------------------------------------------ #

    async def publish_tracks(
        self,
        video_track: Optional[MediaStreamTrack] = None,
        audio_track: Optional[MediaStreamTrack] = None,
        video_name: str = "aivatar-video",
        audio_name: str = "aivatar-audio",
    ) -> None:
        """Publish video and/or audio tracks to the room."""
        _log("[aiortc] publish_tracks called: video=%s audio=%s",
             video_track is not None, audio_track is not None)

        await asyncio.wait_for(self._pc_ready.wait(), timeout=5.0)
        if self._publisher_pc is None:
            raise ConnectionError("Publisher PeerConnection is not initialized")

        tracks_to_add = []

        if video_track is not None:
            cid = str(uuid.uuid4())
            video_track._id = cid  # MATCH track ID with LiveKit cid for MSID
            _log("[aiortc] Generated video cid=%s", cid)

            # Add a sendonly transceiver for the published track (matches the
            # real LiveKit client; publisher offers must be sendonly-only).
            self._publisher_pc.addTransceiver(video_track, direction="sendonly")
            _log("[aiortc] Added sendonly video transceiver for cid=%s", cid)
            # Request track publication from the server
            req = lk_rtc.SignalRequest()
            req.add_track.cid = cid
            req.add_track.name = video_name
            req.add_track.type = lk_models.TrackType.VIDEO
            req.add_track.width = 512
            req.add_track.height = 512
            req.add_track.source = lk_models.TrackSource.CAMERA
            fut = asyncio.get_event_loop().create_future()
            self._track_published_futures[cid] = fut
            await self._send_signal(req)
            _log("[aiortc] Sent add_track(video) cid=%s", cid)
            tracks_to_add.append(("video", cid, fut))

        if audio_track is not None:
            cid = str(uuid.uuid4())
            audio_track._id = cid  # MATCH track ID with LiveKit cid for MSID
            _log("[aiortc] Generated audio cid=%s", cid)

            self._publisher_pc.addTransceiver(audio_track, direction="sendonly")
            _log("[aiortc] Added sendonly audio transceiver for cid=%s", cid)
            req = lk_rtc.SignalRequest()
            req.add_track.cid = cid
            req.add_track.name = audio_name
            req.add_track.type = lk_models.TrackType.AUDIO
            req.add_track.source = lk_models.TrackSource.MICROPHONE
            fut = asyncio.get_event_loop().create_future()
            self._track_published_futures[cid] = fut
            await self._send_signal(req)
            _log("[aiortc] Sent add_track(audio) cid=%s", cid)
            tracks_to_add.append(("audio", cid, fut))

        # Wait for track_published responses
        for kind, cid, fut in tracks_to_add:
            try:
                result = await asyncio.wait_for(fut, timeout=10.0)
                _log("[aiortc] track_published: %s cid=%s sid=%s",
                     kind, cid, result.track.sid)
            except asyncio.TimeoutError:
                _log("[aiortc] TIMEOUT waiting for track_published: %s cid=%s", kind, cid)
                # Continue anyway — the server might process them with the offer

        _log("[aiortc] Creating publisher offer...")
        self._publisher_answer_received.clear()
        self._publisher_offer_id += 1
        offer_id = self._publisher_offer_id
        offer = await self._publisher_pc.createOffer()
        await self._publisher_pc.setLocalDescription(offer)
        _log("[aiortc] Publisher offer SDP summary: %s", _summarize_sdp(offer.sdp))
        if "m=application" not in offer.sdp:
            _log("[aiortc] NOTE: publisher offer has NO DataChannel "
                 "(m=application) section — publisher datachannels are "
                 "disabled to avoid aiortc BUNDLE ICE credential mismatch")
        # Diagnostic: log msid and direction per m-line to verify track matching
        for _line in offer.sdp.splitlines():
            if _line.startswith("a=msid:") or _line.startswith("a=send") or _line.startswith("a=recv"):
                _log("[aiortc] DIAGNOSTIC: %s", _line)
        # Full SDP dump for debugging STATE_MISMATCH
        _log("[aiortc] DIAGNOSTIC: === FULL SDP START (%d bytes) ===", len(offer.sdp))
        for _line in offer.sdp.splitlines():
            _log("[aiortc] DIAGNOSTIC: SDP | %s", _line)
        _log("[aiortc] DIAGNOSTIC: === FULL SDP END ===")

        req = lk_rtc.SignalRequest()
        req.offer.type = "offer"
        req.offer.sdp = self._publisher_pc.localDescription.sdp
        req.offer.id = offer_id
        await self._send_signal(req)
        self._publisher_offer_sent_at = time.monotonic()
        _log("[aiortc] Publisher offer sent (id=%d), waiting for answer...", offer_id)

        # Wait for the publisher answer
        try:
            await asyncio.wait_for(self._publisher_answer_received.wait(), timeout=15.0)
            if self._connection_state == "disconnected":
                if self.reconnect_url:
                    raise LiveKitReconnectException(self.reconnect_url, reconnect=True)
                raise ConnectionError(
                    f"Server disconnected before publisher answer "
                    f"(pc=ice:{self._publisher_pc.iceConnectionState if self._publisher_pc else 'none'})")
            _log("[aiortc] Publisher answer received — media should flow")
        except asyncio.TimeoutError:
            if self.reconnect_url:
                raise LiveKitReconnectException(self.reconnect_url, reconnect=True)
            _log("[aiortc] Publisher answer NOT received within 15s")
            _log("[aiortc] PC state: %s",
                          self._publisher_pc.connectionState if self._publisher_pc else "None")

    # ------------------------------------------------------------------ #
    # Disconnect
    # ------------------------------------------------------------------ #

    async def disconnect(self) -> None:
        """Disconnect from the room."""
        self._connection_state = "disconnected"

        if self._ping_task:
            self._ping_task.cancel()
        if self._listen_task:
            self._listen_task.cancel()

        for pc in (self._publisher_pc, self._subscriber_pc):
            if pc is None:
                continue
            try:
                transport = getattr(pc, "_RTCPeerConnection__transport", None)
                if transport is not None:
                    transport.stop()
            except Exception:
                pass
            await pc.close()

        if self._ws:
            # Send leave request
            try:
                req = lk_rtc.SignalRequest()
                req.leave.SetInParent()
                await self._send_signal(req)
            except Exception:
                pass
            await self._ws.close()

        _logger.info("[aiortc] Disconnected from LiveKit room")

    # ------------------------------------------------------------------ #
    # Internal: signaling
    # ------------------------------------------------------------------ #

    def _build_ws_url(self, url: str, token: str, reconnect: bool = False) -> str:
        """Build the signaling WebSocket URL."""
        parsed = urlparse(url)
        scheme = "wss" if parsed.scheme in ("wss", "https") else "ws"
        host = parsed.hostname
        port = parsed.port

        # Use signaling protocol 8 to match the known-working aiortc LiveKit
        # client (dguerizec/livekit-client-sdk-python). Protocol 13 enables the
        # server's newer fast_publish negotiation flow, which rejects our
        # aiortc publisher offer with NegotiateFailed -> STATE_MISMATCH within
        # ~3ms. Protocol 8 keeps the older, aiortc-compatible publisher flow.
        params_dict = {
            "access_token": token,
            "auto_subscribe": "1",
            "protocol": "8",
            "sdk": "go",
            "version": "1.0.3",
        }
        if reconnect:
            params_dict["reconnect"] = "1"
        params = urlencode(params_dict)

        if port:
            return f"{scheme}://{host}:{port}/rtc?{params}"
        return f"{scheme}://{host}/rtc?{params}"

    async def _send_signal(self, req: lk_rtc.SignalRequest) -> None:
        """Send a SignalRequest over the WebSocket."""
        if self._ws is None:
            return
        data = req.SerializeToString()
        await self._ws.send(data)

    async def _listen_loop(self) -> None:
        """Listen for SignalResponse messages from the server."""
        try:
            async for raw in self._ws:
                data = raw if isinstance(raw, bytes) else raw.encode()
                resp = lk_rtc.SignalResponse()
                resp.ParseFromString(data)
                msg_type = resp.WhichOneof("message")
                _log("[aiortc] <<< SignalResponse: %s", msg_type)
                try:
                    await self._handle_signal(msg_type, resp)
                except Exception as exc:
                    _log("[aiortc] Error handling signal %s: %s\n%s",
                         msg_type, exc, traceback.format_exc())
        except websockets.ConnectionClosed as exc:
            _logger.info("[aiortc] Signaling WebSocket closed: %s", exc)
            self._connection_state = "disconnected"
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _log("[aiortc] Listen loop error: %s\n%s", exc, traceback.format_exc())
            self._connection_state = "disconnected"

    async def _handle_signal(self, msg_type: str, resp: lk_rtc.SignalResponse) -> None:
        """Route a SignalResponse to the appropriate handler."""
        if msg_type == "offer":
            # Server sends an offer for the subscriber PC
            await self._handle_subscriber_offer(resp.offer)

        elif msg_type == "answer":
            # Server sends an answer for the publisher PC
            await self._handle_publisher_answer(resp.answer)

        elif msg_type == "trickle":
            await self._handle_trickle(resp.trickle)

        elif msg_type == "track_published":
            self._handle_track_published(resp.track_published)

        elif msg_type == "leave":
            leave = resp.leave
            reason_name = _enum_name(lk_models.DisconnectReason, leave.reason)
            action_name = _enum_name(lk_rtc.LeaveRequest.Action, leave.action)
            region_list = getattr(getattr(leave, "regions", None), "regions", []) or []
            _log("[aiortc] Server LEAVE: reason=%s action=%s can_reconnect=%s "
                 "regions=%d closest=%s",
                 reason_name, action_name, leave.can_reconnect,
                 len(region_list),
                 region_list[0].region if region_list else "none")
            elapsed = (round(time.monotonic() - self._publisher_offer_sent_at, 3)
                       if self._publisher_offer_sent_at is not None else None)
            _log("[aiortc] LEAVE %ss after publisher offer; "
                 "publisher_answer_received=%s publisher_pc=ice:%s/conn:%s",
                 elapsed,
                 self._publisher_answer_received.is_set(),
                 self._publisher_pc.iceConnectionState if self._publisher_pc else "none",
                 self._publisher_pc.connectionState if self._publisher_pc else "none")
            self._connection_state = "disconnected"
            # action=RECONNECT means reconnect to the SAME URL, not to a
            # region URL (regions are fallback metadata, not redirect targets).
            # Only set reconnect_url if the server actually allows it.
            if leave.action == lk_rtc.LeaveRequest.RECONNECT and leave.can_reconnect:
                self.reconnect_url = self._original_url
                _log("[aiortc] Will reconnect to same URL: %s", self._original_url)
            else:
                self.reconnect_url = None
            self._publisher_answer_received.set()

        elif msg_type in ("pong", "pong_resp"):
            pass  # Ping/pong keepalive

        elif msg_type == "update":
            _logger.debug("[aiortc] Participant update received")

        elif msg_type == "speakers_changed":
            pass

        elif msg_type == "connection_quality":
            pass

        elif msg_type == "room_update":
            pass

        elif msg_type == "stream_state_update":
            pass

        elif msg_type == "subscribed_quality_update":
            pass

        elif msg_type == "refresh_token":
            _log("[aiortc] Server requested token refresh (ignored — short-lived sessions)")

        else:
            _logger.info("[aiortc] Unhandled signal: %s", msg_type)

    async def _handle_subscriber_offer(self, offer: lk_rtc.SessionDescription) -> None:
        """Handle an SDP offer for the subscriber PeerConnection."""
        if self._subscriber_pc is None:
            raise ConnectionError("Subscriber PeerConnection is not initialized")
        _log("[aiortc] Received subscriber offer (type=%s, id=%d, sdp=%d bytes)",
             offer.type, offer.id, len(offer.sdp))

        sdp = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        await self._subscriber_pc.setRemoteDescription(sdp)
        _log("[aiortc] Subscriber remoteDescription set")

        pending = self._pending_candidates[lk_rtc.SignalTarget.SUBSCRIBER]
        if pending:
            _logger.info("[aiortc] Applying %d pending ICE candidates",
                         len(pending))
        for cand_data in pending:
            await self._apply_ice_candidate(self._subscriber_pc, cand_data)
        pending.clear()

        answer = await self._subscriber_pc.createAnswer()
        await self._subscriber_pc.setLocalDescription(answer)
        _log("[aiortc] Subscriber answer created (%d bytes)", len(answer.sdp))

        req = lk_rtc.SignalRequest()
        req.answer.type = "answer"
        req.answer.sdp = self._subscriber_pc.localDescription.sdp
        req.answer.id = offer.id
        await self._send_signal(req)
        _log("[aiortc] >>> Subscriber answer sent to server (id=%d)", offer.id)

        self._subscriber_answer_sent.set()

    async def _handle_publisher_answer(self, answer: lk_rtc.SessionDescription) -> None:
        """Handle an SDP answer for the publisher PeerConnection."""
        if self._publisher_pc is None:
            raise ConnectionError("Publisher PeerConnection is not initialized")
        _log("[aiortc] Received publisher answer (type=%s, id=%d, sdp=%d bytes)",
             answer.type, answer.id, len(answer.sdp))

        if answer.id and answer.id != self._publisher_offer_id:
            _logger.warning(
                "[aiortc] Ignoring publisher answer for old offer id=%d latest=%d",
                answer.id,
                self._publisher_offer_id,
            )
            return

        sdp = RTCSessionDescription(sdp=answer.sdp, type=answer.type)
        await self._publisher_pc.setRemoteDescription(sdp)
        _log("[aiortc] Publisher remoteDescription set")

        pending = self._pending_candidates[lk_rtc.SignalTarget.PUBLISHER]
        if pending:
            _logger.info("[aiortc] Applying %d pending ICE candidates",
                         len(pending))
        for cand_data in pending:
            await self._apply_ice_candidate(self._publisher_pc, cand_data)
        pending.clear()

        self._publisher_answer_received.set()

    async def _handle_trickle(self, trickle: lk_rtc.TrickleRequest) -> None:
        """Handle an ICE candidate from the server."""
        try:
            cand_init = json.loads(trickle.candidateInit)
        except (json.JSONDecodeError, ValueError) as exc:
            _logger.warning("[aiortc] Invalid ICE candidate JSON: %s", exc)
            return

        target = trickle.target
        target_name = "PUBLISHER" if target == lk_rtc.SignalTarget.PUBLISHER else "SUBSCRIBER"
        candidate_str = cand_init.get("candidate", "")
        _logger.debug("[aiortc] ICE candidate for %s: %s", target_name, candidate_str[:80])

        pc = (
            self._publisher_pc
            if target == lk_rtc.SignalTarget.PUBLISHER
            else self._subscriber_pc
        )
        if pc and pc.remoteDescription:
            await self._apply_ice_candidate(pc, cand_init)
        else:
            self._pending_candidates.setdefault(target, []).append(cand_init)

    async def _apply_ice_candidate(self, pc: RTCPeerConnection, cand_init: dict) -> None:
        """Apply an ICE candidate to a PeerConnection."""
        candidate_str = cand_init.get("candidate", "")
        if not candidate_str:
            return  # End-of-candidates signal

        sdp_mid = cand_init.get("sdpMid", "")
        sdp_mline_index = cand_init.get("sdpMLineIndex", 0)

        try:
            candidate = candidate_from_sdp(candidate_str)
            candidate.sdpMid = sdp_mid
            candidate.sdpMLineIndex = sdp_mline_index
            await pc.addIceCandidate(candidate)
        except Exception as exc:
            _logger.debug("[aiortc] Failed to add ICE candidate: %s — %s",
                          candidate_str[:60], exc)

    def _handle_track_published(self, msg: lk_rtc.TrackPublishedResponse) -> None:
        """Handle track_published response."""
        cid = msg.cid
        _log("[aiortc] track_published response: cid=%s track_sid=%s", cid, msg.track.sid)
        fut = self._track_published_futures.pop(cid, None)
        if fut and not fut.done():
            fut.set_result(msg)
        else:
            _logger.warning("[aiortc] No pending future for cid=%s (already resolved?)", cid)

    # ------------------------------------------------------------------ #
    # Internal: PeerConnection setup
    # ------------------------------------------------------------------ #

    def _setup_pc(self, pc: RTCPeerConnection, target: int, label: str) -> None:
        """Configure event handlers on a PeerConnection."""

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            _log("[aiortc] %s PC received track: kind=%s id=%s", label, track.kind, track.id)
            if self._on_track_subscribed:
                self._on_track_subscribed(track)

        @pc.on("icecandidate")
        def on_ice_candidate(candidate: RTCIceCandidate):
            if candidate is None:
                return
            asyncio.ensure_future(self._send_ice_candidate(
                candidate, target
            ))

        @pc.on("connectionstatechange")
        def on_state_change():
            _log("[aiortc] %s PC connectionState: %s", label, pc.connectionState)

        @pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            _log("[aiortc] %s PC iceConnectionState: %s", label, pc.iceConnectionState)

        @pc.on("icegatheringstatechange")
        def on_ice_gathering():
            _logger.info("[aiortc] %s PC iceGatheringState: %s", label, pc.iceGatheringState)

    async def _send_ice_candidate(self, candidate: RTCIceCandidate, target: int) -> None:
        """Send a local ICE candidate to the server."""
        cand_init = {
            "candidate": f"candidate:{candidate.foundation} {candidate.component} "
                         f"{candidate.protocol} {candidate.priority} "
                         f"{candidate.ip} {candidate.port} typ {candidate.type}",
            "sdpMid": candidate.sdpMid or "",
            "sdpMLineIndex": candidate.sdpMLineIndex or 0,
        }

        req = lk_rtc.SignalRequest()
        req.trickle.candidateInit = json.dumps(cand_init)
        req.trickle.target = target
        await self._send_signal(req)

    # ------------------------------------------------------------------ #
    # Internal: keepalive
    # ------------------------------------------------------------------ #

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        try:
            interval = 10
            if self._join_response and self._join_response.ping_interval > 0:
                interval = self._join_response.ping_interval
            while True:
                await asyncio.sleep(interval)
                if self._connection_state != "connected":
                    break
                req = lk_rtc.SignalRequest()
                req.ping = int(asyncio.get_event_loop().time() * 1000)
                await self._send_signal(req)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _logger.debug("[aiortc] Ping loop error: %s", exc)
