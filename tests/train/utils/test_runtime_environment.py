from skyrl.train.config.config import SkyRLTrainConfig
from skyrl.train.utils import utils as train_utils


def test_prepare_runtime_environment_exports_remote_sandbox_config(monkeypatch):
    expected = {
        "ALLHANDS_API_KEY": "allhands-key",
        "SANDBOX_API_KEY": "sandbox-key",
        "SANDBOX_REMOTE_RUNTIME_API_URL": "http://runtime:3000",
        "OPENHANDS_FILE_STORE_PATH": "/data/tovi/openhands_file_store",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: True)

    env_vars = train_utils.prepare_runtime_environment(SkyRLTrainConfig())

    assert expected.items() <= env_vars.items()
