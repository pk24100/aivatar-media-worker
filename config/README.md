Optional worker config files live here.

`default_avatars.manifest.json` is intentionally not checked in.
If you add it before production, the Modal worker will preload the listed
default avatars during `@modal.enter(snap=True)` and reuse them on matching
session `sourceImage` URLs. If the file is absent, the worker falls back to
the normal per-session URL fetch path.

For fixed platform defaults only, the same manifest can snapshot compressed
idle-video bytes. Custom avatar idle clips are always fetched per session.

```json
{
  "avatars": [
    {
      "id": "default-avatar-id",
      "matchUrls": ["https://avatars.facemode.io/avatars/default/example.png"],
      "idleVideo": {
        "key": "default-avatar-id/v1/example.mp4"
      }
    }
  ]
}
```

Provide `IDLE_VIDEO_R2_ENDPOINT`, `IDLE_VIDEO_R2_ACCESS_KEY_ID`, and
`IDLE_VIDEO_R2_SECRET_ACCESS_KEY` in the Modal worker secret before deploy.
The worker later matches sessions by immutable `key` and uses the
snapshotted MP4 bytes.

The public default-bucket key is both the FaceMode React SDK bootstrap asset
and the worker snapshot asset. Configure the worker's read-only R2 credential
for `facemode-idle-videos-default`. Custom `idle/custom/...` keys never belong
in this manifest and never reach a browser.
