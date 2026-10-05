import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_only(device):
    if device.type != "cuda":
        pytest.skip("CUDA not available")
