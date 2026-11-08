#!/usr/bin/env python3
"""Install pinned baseline files from a checkout the user is authorized to use.

No repository download occurs. Nothing installed by this utility should be
committed or redistributed with the PhysTok candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def install(model: str, source_dir: Path, *, confirm_rights: bool = False) -> dict:
    if model not in ('timexer', 'xlinear'):
        raise ValueError('Only the two non-redistributed baseline sources are supported')
    if not confirm_rights:
        raise PermissionError('Confirm your right to use this source with --confirm-source-use-rights; public visibility alone is not a license')
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    specs = [x for x in json.loads((ROOT/'baselines/EXTERNAL_SOURCES.json').read_text())['files'] if x['model'] == model]
    if (source_dir/'.git').exists():
        head = subprocess.check_output(['git','-C',str(source_dir),'rev-parse','HEAD'],text=True).strip()
        if head != specs[0]['commit']:
            raise ValueError('Checkout HEAD differs from pinned source commit')
    destination = ROOT/'phystok/models/vendor'/model
    if destination.exists():
        raise FileExistsError('An external installation already exists; inspect it before attempting replacement')
    # Validate every source and transformation before writing any file.
    prepared = []
    for spec in specs:
        source = source_dir/spec['upstream_path']
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec['original_sha256']:
            raise ValueError(f"Pinned source hash mismatch: {spec['upstream_path']}")
        text = raw.decode()
        for replacement in spec['replacements']:
            if replacement['old'] not in text:
                raise ValueError('Expected adaptation site is absent')
            text = text.replace(replacement['old'],replacement['new'])
        runtime = text.encode()
        if hashlib.sha256(runtime).hexdigest() != spec['public_runtime_sha256']:
            raise ValueError('Adapted implementation does not match frozen benchmark math/imports')
        prepared.append((spec['upstream_path'],runtime))
    for name, raw in prepared:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    for folder in [destination,*[x for x in destination.rglob('*') if x.is_dir()]]:
        (folder/'__init__.py').touch()
    result = {'status':'PASS','model':model,'commit':specs[0]['commit'],
              'files':len(prepared),'installed_path':str(destination.relative_to(ROOT)),
              'redistribution_permission':'UNCONFIRMED: do not commit or redistribute this installation'}
    (destination/'LOCAL_INSTALLATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['timexer','xlinear'],required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--confirm-source-use-rights',action='store_true')
    args=parser.parse_args()
    print(json.dumps(install(args.model,args.source_dir,confirm_rights=args.confirm_source_use_rights)))


if __name__ == '__main__':
    main()
