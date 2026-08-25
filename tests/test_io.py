import json

import cv2
import numpy as np

from liquid_depth.io import load_frame


def test_load_frame(tmp_path):
    frame = tmp_path / "frame_001"
    frame.mkdir()
    cv2.imwrite(str(frame / "rgb.png"), np.zeros((3, 4, 3), np.uint8))
    np.save(frame / "depth.npy", np.ones((3, 4), np.uint16))
    info = {"K": [2, 0, 1, 0, 2, 1, 0, 0, 1]}
    (frame / "depth_info.json").write_text(json.dumps(info), encoding="utf-8")
    loaded = load_frame(frame)
    assert loaded.frame_id == "frame_001"
    assert loaded.depth.shape == (3, 4)
    assert loaded.camera_matrix[0, 0] == 2

