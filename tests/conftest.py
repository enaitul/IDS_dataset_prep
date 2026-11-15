import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.fixtures import make_workspace
from src import pipeline

@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    """Full pipeline run on TEST-ONLY synthetic fixtures (all three schemas)."""
    tmp = tmp_path_factory.mktemp("ws")
    cfg = make_workspace(tmp)
    pipeline.run(cfg)
    return tmp, cfg
