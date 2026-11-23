import json
import pytest
from scripts.prepare_registry import prepare_registry


def test_merge_registry_locations_and_reject_conflicts(tmp_path):
    (tmp_path / 'registry').mkdir()
    mace = tmp_path / 'registry_mace.json'
    mace.write_text(json.dumps({'mace': {'backend': 'mace'}}))
    nep = tmp_path / 'registry/registry_nep.json'
    nep.write_text(json.dumps({'nep89': {'backend': 'nep'}}))
    out = tmp_path / 'combined.json'
    assert prepare_registry(tmp_path, out) == out
    assert set(json.loads(out.read_text())) == {'mace', 'nep89'}
    nep.write_text(json.dumps({'mace': {'backend': 'nep'}}))
    with pytest.raises(ValueError, match='Conflicting potential'):
        prepare_registry(tmp_path, out)
    assert prepare_registry(tmp_path, out, str(mace)) == mace


def test_missing_registry(tmp_path):
    with pytest.raises(ValueError, match='No local potential registry'):
        prepare_registry(tmp_path, tmp_path / 'out.json')
