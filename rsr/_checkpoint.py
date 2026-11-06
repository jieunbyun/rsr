"""Write individual reference checkpoints without truncating the last good copy."""

import json
import os
import tempfile
import time

import torch


def _replace_with_retry(source, destination):
    # Windows readers/scanners can briefly deny replacement (including access
    # denied). Retry only these errors, for at most about eight seconds.
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 9:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))


def _atomic_save(path, write, *, binary):
    path = os.path.abspath(os.fspath(path))
    # Keep the temporary file on the same filesystem as its destination. Use
    # the open handle for serialization so torch never reopens a Windows path.
    fd, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix="." + os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        kwargs = {} if binary else {"encoding": "utf-8"}
        with os.fdopen(fd, "wb" if binary else "w", **kwargs) as stream:
            write(stream)
        _replace_with_retry(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            # Replacement removes the temporary path. Cleanup failures must
            # also not hide the original serialization/replacement error.
            pass


def save_json(obj, path):
    _atomic_save(path, lambda stream: json.dump(obj, stream, indent=4), binary=False)


def save_tensor(tensor, path):
    tensor = tensor.detach().cpu()
    _atomic_save(path, lambda stream: torch.save(tensor, stream), binary=True)
