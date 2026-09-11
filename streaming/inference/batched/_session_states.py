"""Session state constants and timing configuration."""

import os

# =============================================================================
# Session States
# =============================================================================
INITIALIZING = "INITIALIZING"
ACTIVE = "ACTIVE"
DRAINING = "DRAINING"
ENDED = "ENDED"

# --- Config ---
WAIT_WINDOW_MS = 20
IDLE_TIMEOUT_S = 10
REACTIVATION_TIMEOUT_S = int(os.environ.get("AIVATAR_REACTIVATION_TIMEOUT_S", "120"))
