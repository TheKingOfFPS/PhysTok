from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
MODELS = ('phystok', 'dlinear', 'informer', 'patchtst', 'itransformer', 'timexer', 'xlinear')
SEEDS = (20260826, 20260827, 20260828)
SCENARIOS = ('Spatial', 'Future', 'Joint', 'External-Spatial', 'External-Joint')


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def canonical_gpu_uuid(value):
    """PyTorch 2.7 exposes a bare UUID; NVML/nvidia-smi require GPU- prefix."""
    text=str(value)
    bare=text[4:] if text.startswith('GPU-') else text
    require(re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',bare),
            'unrecognized GPU UUID representation')
    return 'GPU-'+bare.lower()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def fingerprint(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_output(path, root=None):
    p = Path(path).absolute()
    for q in [p, *p.parents]:
        require(not q.is_symlink(), f'symlink output forbidden: {q}')
    if root is not None:
        root = Path(root).resolve()
        require(p.resolve().is_relative_to(root) and p.resolve()!=root, 'output outside specified root')
    require(p != Path('/'), 'root output forbidden')
    return p


def atomic_bytes(path, data, *, replace=False, root=None):
    path = safe_output(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            # Atomic exclusive publish. Existing artifacts are never overwritten.
            os.link(tmp, path)
            os.unlink(tmp)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_json(path, value, **kwargs):
    return atomic_bytes(path, canonical(value) + b'\n', **kwargs)


def slug(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value), 'invalid artifact identity')
    return value


def assert_identity(actual, expected):
    require(actual == expected, f'checkpoint/run identity mismatch: {actual} != {expected}')


def verify_artifacts(manifest_path, *, expected_identity=None):
    p = Path(manifest_path)
    doc = read_json(p)
    if expected_identity is not None:
        assert_identity(doc['identity'], expected_identity)
    require(doc['status'] == 'SUCCEEDED' and doc['artifacts'], 'no verified success artifacts')
    for rel, digest in doc['artifacts'].items():
        item = (p.parent / rel).resolve()
        require(item.is_relative_to(p.parent.resolve()) and not (p.parent / rel).is_symlink(), 'artifact path escape')
        require(item.is_file() and sha256(item) == digest, f'missing/tampered artifact: {rel}')
    return doc
