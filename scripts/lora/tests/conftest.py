"""pytest configuration for scripts/lora/tests.

Session-scoped fixtures that apply to all tests in this package.
"""
import pytest
import torch
import torch._dynamo


@pytest.fixture(scope="session", autouse=True)
def suppress_torch_dynamo_errors():
    """Suppress Torch Inductor compilation errors on Windows (no MSVC cl.exe).

    Session-scoped autouse: applies to all tests without leaking a module-level
    side-effect. dynamo falls back to eager execution automatically.
    """
    torch._dynamo.config.suppress_errors = True
