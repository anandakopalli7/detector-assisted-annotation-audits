"""Explicit paths, immutable chunks, checksums and an OS-held worker lock."""
import gzip,hashlib,json,os,sys,platform
from pathlib import Path
from datetime import datetime,timezone

def utc():return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return sha_bytes(Path(p).read_bytes())
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def owned(path,root):
    p=Path(path).resolve();r=Path(root).resolve()
    if p==r or not p.is_relative_to(r):raise ValueError('output must be strictly inside approved research root')
    if any(('restricted' in s.lower() or 'sealed' in s.lower()) for s in p.relative_to(r).parts):raise ValueError('closed path')
    return p

def new_json(path,obj):
    data=canonical(obj)
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())

class WorkerLock:
    """OS byte lock survives launcher/child topology and releases on death."""
    def __init__(self,path):self.path=Path(path);self.f=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.f=self.path.open('a+b')
        if self.path.stat().st_size==0:self.f.write(b'0');self.f.flush()
        self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.f.close();self.f=None
            raise RuntimeError('another worker owns the research lock')
        return self
    def __exit__(self,*args):
        if self.f:
            self.f.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_UN)
            self.f.close()

def write_chunk(path,payload):
    path=Path(path);tmp=path.with_name(path.name+'.partial');side=path.with_name(path.name+'.sha256.json')
    if path.exists() or tmp.exists() or side.exists():raise FileExistsError('chunk or partial already exists')
    data=gzip.compress(canonical(payload),mtime=0)
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)
    new_json(side,{'sha256':sha_bytes(data),'context_sha256':payload['context_sha256']})

def read_chunk(path,context):
    path=Path(path);side=load(path.with_name(path.name+'.sha256.json'));data=path.read_bytes()
    if side['sha256']!=sha_bytes(data) or side['context_sha256']!=context:raise ValueError('chunk checksum/context mismatch')
    obj=json.loads(gzip.decompress(data))
    if obj['context_sha256']!=context:raise ValueError('chunk payload context')
    return obj

def package_runtime():
    import importlib.metadata as md
    import numpy as np
    import scipy
    import io,contextlib
    versions={'numpy':np.__version__,'scipy':scipy.__version__}
    package_files={}
    for name in versions:
        dist=md.distribution(name);found={}
        for rel in dist.files or []:
            if str(rel).endswith('.pyc') or '__pycache__' in str(rel):continue
            p=Path(dist.locate_file(rel))
            if p.is_file():found[str(rel).replace('\\','/')]=file_sha(p)
        package_files[name]=found
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):np.show_config()
    rng=np.random.Generator(np.random.PCG64(123456789))
    return {'python_version':platform.python_version(),'python_binary_sha256':file_sha(sys.executable),
            'base_python_binary_sha256':file_sha(getattr(sys,'_base_executable',sys.executable)),
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'platform':platform.platform(),'packages':versions,'package_files':package_files,
            'blas_config':buf.getvalue(),'pcg64_fixture':rng.integers(0,2**32,size=16,dtype=np.uint64).tolist()}

def validate_manifest(path,check_runtime=True):
    path=Path(path).resolve();manifest=load(path);base=path.parent
    if manifest.get('schema')!='PAPER2_COMPARATOR_RELEASE_V1':raise ValueError('manifest schema')
    for rel,h in manifest['files'].items():
        p=(base/rel).resolve()
        if not p.is_relative_to(base):raise ValueError('manifest path escape')
        if file_sha(p)!=h:raise ValueError('release hash changed: '+rel)
    config=load(base/manifest['config'])
    from .methods import METHODS
    if config['legacy_methods']+config['new_methods']!=list(METHODS):raise ValueError('method identity')
    if config['fold_seed_prefix']!='P2_COMPARE_FRAME_FOLDS_V01':raise ValueError('fold namespace mismatch')
    code={str(p.relative_to(base)).replace('\\','/') for p in (base/'paper2_compare').rglob('*.py')}
    if not code.issubset(manifest['files']):raise ValueError('unindexed executable source')
    if manifest['config'] not in manifest['files'] or manifest['runtime'] not in manifest['files']:raise ValueError('unbound config/runtime')
    if len(config['legacy_methods']+config['new_methods'])!=13 or config['original_medium_verdict']!='NOT_ATTAINED':raise ValueError('config scope')
    if check_runtime:
        actual=package_runtime();expected=load(base/manifest['runtime'])
        if actual!=expected:raise ValueError('runtime identity changed')
    return manifest,config

def execution_permission(manifest_path,approval_path,output,root):
    manifest_path=Path(manifest_path).resolve()
    # Check consent before any dataset construction or expensive runtime scan.
    receipt=load(approval_path)
    if receipt.get('decision')!='D-P2-13' or receipt.get('status')!='APPROVED_FIXED_COMPARATOR_RUN':raise PermissionError('separate D-P2-13 execution approval required')
    if receipt.get('manifest_sha256')!=file_sha(manifest_path):raise PermissionError('approval does not bind this release')
    if not receipt.get('exact_user_reply'):raise PermissionError('approval source missing')
    if Path(receipt['research_root']).resolve()!=Path(root).resolve():raise PermissionError('root approval mismatch')
    target=owned(output,root)
    if target!=owned(receipt['output_directory'],root):raise PermissionError('output approval mismatch')
    manifest,config=validate_manifest(manifest_path)
    return target,manifest,config
