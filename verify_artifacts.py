"""Read-only source/data integrity checks; no experiments or network requests."""
from pathlib import Path
import argparse,hashlib,json,math
ROOT=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check(code_only=False):
 m=json.loads((ROOT/'CODE_MANIFEST.json').read_text())
 for name,h in m.items():
  p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.is_file(),name
  assert digest(p)==h,name
 count=len(m)
 if not code_only:
  for manifest in ['INPUTS_MANIFEST.json','RESULTS_MANIFEST.json']:
   for name,h in json.loads((ROOT/manifest).read_text()).items():
    p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.is_file(),name
    assert digest(p)==h,name
    count+=1
  original=json.loads((ROOT/'03_protocol/EXECUTION_MANIFEST.json').read_text())
  omitted=json.loads((ROOT/'03_protocol/RELEASE_SCOPE.json').read_text())['omitted_original_sources']
  for name,h in original['sources'].items():
   if name not in omitted:assert digest(ROOT/name)==h,name
  s=json.loads((ROOT/'05_results/ANALYSIS_V2/SUMMARY.json').read_text())
  assert s['paths']==13000 and len(s['cells'])==39 and len(s['coverage_family'])==117
  for row in s['primary_contrasts']:
   n=row['n'];v=row['mcse']**2*n;l=math.log(480)
   radius=math.sqrt(2*v*l/n)+7*40000*l/(3*(n-1))
   assert math.isclose(row['simultaneous_finite95'][0],max(-20000,row['difference']-radius),abs_tol=1e-9)
   assert math.isclose(row['simultaneous_finite95'][1],min(20000,row['difference']+radius),abs_tol=1e-9)
 return {'status':'PASS','files_checked':count,'mode':'code only' if code_only else 'code, inputs and aggregate results','performance_experiments':0}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-only',action='store_true');a=p.parse_args();print(json.dumps(check(a.code_only)))
