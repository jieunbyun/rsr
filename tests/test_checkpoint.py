import errno
import json
import os

import pytest
import torch

from rsr import _checkpoint


def test_checkpoint_round_trip(tmp_path):
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    tensor_path = tmp_path / "refs.pt"
    json_path = tmp_path / "refs.json"
    _checkpoint.save_tensor(torch.zeros(1), tensor_path)
    _checkpoint.save_tensor(tensor, tensor_path)
    _checkpoint.save_json({"e01": [">=", 2]}, json_path)
    loaded = torch.load(tensor_path, weights_only=True)
    assert torch.equal(loaded, tensor)
    assert not loaded.requires_grad
    assert loaded.device.type == "cpu"
    assert json.loads(json_path.read_text()) == {"e01": [">=", 2]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["refs.json", "refs.pt"]


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_temporary_windows_lock_is_retried(tmp_path, monkeypatch, winerror):
    path = tmp_path / "refs.pt"
    _checkpoint.save_tensor(torch.tensor([1]), path)
    replace = _checkpoint.os.replace
    attempts = []
    sleeps = []

    def locked_replace(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            assert torch.equal(torch.load(path, weights_only=True), torch.tensor([1]))
            exc = PermissionError("temporarily locked")
            exc.winerror = winerror
            raise exc
        return replace(source, destination)

    monkeypatch.setattr(_checkpoint.os, "replace", locked_replace)
    monkeypatch.setattr(_checkpoint.time, "sleep", sleeps.append)
    _checkpoint.save_tensor(torch.tensor([2]), path)
    assert len(attempts) == 3
    assert sleeps == [0.05, 0.1]
    assert torch.equal(torch.load(path, weights_only=True), torch.tensor([2]))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("winerror, expected_attempts", [(32, 10), (None, 1)])
def test_replace_failure_preserves_checkpoint(tmp_path, monkeypatch, winerror, expected_attempts):
    path = tmp_path / "refs.json"
    _checkpoint.save_json({"old": True}, path)
    attempts = []
    exc = OSError(errno.EIO, "replacement failed")
    if winerror is not None:
        exc.winerror = winerror

    def fail_replace(*args):
        attempts.append(args)
        raise exc

    monkeypatch.setattr(_checkpoint.os, "replace", fail_replace)
    monkeypatch.setattr(_checkpoint.time, "sleep", lambda _: None)
    with pytest.raises(OSError) as caught:
        _checkpoint.save_json({"new": True}, path)
    assert caught.value is exc
    assert len(attempts) == expected_attempts
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_preserves_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "refs.pt"
    _checkpoint.save_tensor(torch.tensor([1]), path)

    def fail_save(tensor, stream):
        stream.write(b"partial checkpoint")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(_checkpoint.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failed"):
        _checkpoint.save_tensor(torch.tensor([2]), path)
    assert torch.equal(torch.load(path, weights_only=True), torch.tensor([1]))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_real_windows_reader_lock(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes

    path = tmp_path / "refs.pt"
    _checkpoint.save_tensor(torch.tensor([1]), path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    # A reader that allows other readers, but denies writes and deletion.
    handle = kernel32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    sleeps = []

    def release_lock(delay):
        nonlocal handle
        sleeps.append(delay)
        assert torch.equal(torch.load(path, weights_only=True), torch.tensor([1]))
        assert kernel32.CloseHandle(handle)
        handle = None

    monkeypatch.setattr(_checkpoint.time, "sleep", release_lock)
    try:
        _checkpoint.save_tensor(torch.tensor([2]), path)
    finally:
        if handle is not None:
            kernel32.CloseHandle(handle)
    assert sleeps == [0.05]
    assert torch.equal(torch.load(path, weights_only=True), torch.tensor([2]))
