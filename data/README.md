# Runtime data

Large RGB-D recordings are intentionally not versioned in Git. On the configured server they live at:

```text
/root/autodl-tmp/liquid-depth-data/
  legacy_rgbd/                 # migrated output/<frame_id> recordings
  hardware/                    # archived Orbbec SDK bundle
  manifests/                   # checksums and dataset inventories
```

Every frame directory uses the following stable contract:

```text
<frame_id>/
  rgb.png
  depth.npy
  color_info.json
  depth_info.json
```

Human-reviewed masks should be stored in a separate versioned dataset release, never overwrite raw captures.

