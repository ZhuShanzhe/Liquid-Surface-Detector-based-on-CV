"""Publication resource validation; never changes measurement algorithms."""
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from measure_release import validated_profile


def test_profile_checks_hash_and_resolves_paths(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'assets').mkdir()
    asset=tmp_path/'assets/a.bin';asset.write_bytes(b'owned-test-model')
    manifest={'models':{'model_checkpoint':{'path':'assets/a.bin','sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}}}
    (tmp_path/'configs/release_assets.json').write_text(json.dumps(manifest))
    assert validated_profile(tmp_path)=={'model_checkpoint':str(asset.resolve())}


def test_profile_rejects_corrupt_checkpoint(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'assets').mkdir()
    (tmp_path/'assets/a.bin').write_bytes(b'bad')
    (tmp_path/'configs/release_assets.json').write_text(json.dumps({'models':{'model_checkpoint':{'path':'assets/a.bin','sha256':'wrong'}}}))
    with pytest.raises(ValueError,match='hash_mismatch'):validated_profile(tmp_path)


def test_profile_rejects_path_escape(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/release_assets.json').write_text(json.dumps({'models':{'model_checkpoint':{'path':'../outside.bin','sha256':'wrong'}}}))
    with pytest.raises(ValueError,match='outside_release_root'):validated_profile(tmp_path)
