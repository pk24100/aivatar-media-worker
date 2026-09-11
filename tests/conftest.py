import os

# LiveKit egress connects run a real UDP sendability probe against the cluster
# TURN endpoint by default. Keep tests hermetic: probe-specific tests re-enable
# it with monkeypatched fakes.
os.environ.setdefault("LIVEKIT_NET_PROBE_ENABLED", "false")
