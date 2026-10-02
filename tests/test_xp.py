import builtins
import sys

import numpy

from pk import xp


def test_live_backend_is_numpy():
    assert xp.live is numpy


def test_consolidation_falls_back_to_numpy_without_cupy(monkeypatch):
    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("no cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_cupy)
    xp.consolidation_backend.cache_clear()
    try:
        module, device = xp.consolidation_backend()
    finally:
        xp.consolidation_backend.cache_clear()
    assert module is numpy
    assert device == "cpu"


def test_consolidation_falls_back_when_no_cuda_device(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 0

    class FakeCuda:
        runtime = FakeRuntime

    class FakeCupy:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "cupy", FakeCupy)
    xp.consolidation_backend.cache_clear()
    try:
        module, device = xp.consolidation_backend()
    finally:
        xp.consolidation_backend.cache_clear()
    assert module is numpy
    assert device == "cpu"


def test_consolidation_uses_cupy_when_device_present(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1

    class FakeCuda:
        runtime = FakeRuntime

    class FakeCupy:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "cupy", FakeCupy)
    xp.consolidation_backend.cache_clear()
    try:
        module, device = xp.consolidation_backend()
    finally:
        xp.consolidation_backend.cache_clear()
    assert module is FakeCupy
    assert device == "cuda"


def test_consolidation_backend_is_resolved_once():
    xp.consolidation_backend.cache_clear()
    try:
        assert xp.consolidation_backend() is xp.consolidation_backend()
    finally:
        xp.consolidation_backend.cache_clear()
