"""Verify submission-v1 after extraction. Read-only; no network or cloud."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent.parent
m=json.loads((ROOT/'ARCHIVE_MANIFEST_SUBMISSION.json').read_text())
for n,r in m.items():
 p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve())
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],n
print(json.dumps({'status':'PASS','submission_manifest_members':len(m)}),flush=True)
subprocess.run([sys.executable,str(ROOT/'14_review_response_v01/verify_current_evidence.py')],check=True)
