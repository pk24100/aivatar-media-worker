Optional worker config files live here.

`default_avatars.manifest.json` is intentionally not checked in.
If you add it before production, the Modal worker will preload the listed
default avatars during `@modal.enter(snap=True)` and reuse them on matching
session `sourceImage` URLs. If the file is absent, the worker falls back to
the normal per-session URL fetch path.
