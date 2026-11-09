#!/usr/bin/env python3
"""Inspect actual staged/committed files, not merely .gitignore intentions."""
import argparse,csv,hashlib,io,json,re,subprocess
from pathlib import Path

def inspect(root,commit=None,allowed_remote_urls=()):
    root=Path(root).resolve()
    command=['git','ls-tree','-rz','--name-only',commit] if commit else ['git','ls-files','-z']
    names=subprocess.check_output(command,cwd=root).decode().split('\0')
    names=sorted(n for n in names if n)
    findings=[];inventory=[]
    private_patterns={
        'private_home_path':re.compile('/'+'home'+r'/[A-Za-z0-9_.-]+/'),
        'private_root_path':re.compile('/'+'root'+r'/'),
        'windows_user_path':re.compile(r'[A-Za-z]:\\Users\\[^\\\s]+'),
        'private_ipv4':re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
        'private_key':re.compile(r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'),
        'github_credential':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
        'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
        'credential_assignment':re.compile(r'''(?im)\b(?:password|api_key|access_token|secret_key)\s*[:=]\s*["']([^"'\n]{8,})["']'''),
    }
    banned={'.nc','.nc4','.grib','.grb','.npy','.npz','.parquet','.pt','.pth','.ckpt','.safetensors','.sqlite','.db','.zip','.tar','.gz','.pdf','.log'}
    for name in names:
        mode=['git','show',f'{commit}:{name}' if commit else f':{name}']
        data=subprocess.check_output(mode,cwd=root)
        inventory.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        issues=[]
        path=Path(name)
        if path.suffix.lower() in banned:issues.append('prohibited binary/data/log extension')
        if any(p in {'.env','.ssh','__pycache__','.venv','runs','outputs'} for p in path.parts):issues.append('private/runtime directory')
        if path.name.startswith(('.env','id_rsa','id_ed25519')):issues.append('credential filename')
        if name.startswith(('phystok/models/vendor/timexer/','phystok/models/vendor/xlinear/')):issues.append('unconfirmed-license source')
        if len(data)>2*1024*1024:issues.append('unexpected large asset')
        if (root/name).is_symlink():issues.append('symlink')
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            issues.append('non-text asset');text=''
        for kind,pattern in private_patterns.items():
            if pattern.search(text):issues.append(kind)
        if path.suffix=='.csv':
            if not (name.startswith('results/') or name.startswith('data/metadata/') or name=='PUBLIC_FILE_MANIFEST.csv'):
                issues.append('CSV outside approved aggregate/metadata paths')
            header=next(csv.reader(io.StringIO(text)),[])
            if set(header)&{'y_true','y_pred','truth','prediction','timestamp','issue_time','target_time','value'}:
                issues.append('individual-observation/prediction CSV columns')
        if issues:findings.append({'path':name,'issues':issues})
    remotes=subprocess.check_output(['git','remote'],cwd=root,text=True).splitlines()
    for remote in remotes:
        url=subprocess.check_output(['git','remote','get-url',remote],cwd=root,text=True).strip()
        if url not in set(allowed_remote_urls):
            findings.append({'path':'.git/config','issues':['remote is not in the explicit authorized URL list']})
    return {'status':'PASS' if not findings else 'FAIL','scope':'actual Git '+('commit '+commit if commit else 'index'),
            'files':len(names),'bytes':sum(x['bytes'] for x in inventory),'remote_count':len(remotes),
            'findings':findings,'inventory':inventory,
            'limitations':['Pattern scanning is supplemented by manual source, licensing and aggregate-data review. It does not grant publication rights.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    p.add_argument('--commit');p.add_argument('--output')
    p.add_argument('--allow-remote-url',action='append',default=[],help='Explicit authorized remote URL; defaults to no remote allowed')
    a=p.parse_args();result=inspect(a.root,a.commit,a.allow_remote_url)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)

if __name__=='__main__':main()
