import pytest

from scripts.import_qdrant import ROOT, embedding_cache_path


def test_cache_location_comes_from_the_verified_run_manifest():
    key = 'a' * 64
    assert embedding_cache_path({'cache_key': key}) == ROOT / 'data/embedding_cache' / key
    for invalid in ('../outside', '', 'A' * 64, None):
        with pytest.raises(ValueError):
            embedding_cache_path({'cache_key': invalid})
