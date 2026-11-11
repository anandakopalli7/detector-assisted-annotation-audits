"""Separate arithmetic reduction; does not import the production reducer or runner."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,variance
from scipy.stats import binom
import gzip,json,math,hashlib,time
R=Path(__file__).resolve().parents[1];A=R/'05_results/ANALYSIS_V1'
s=json.loads((A/'SUMMARY.json').read_text());inputs=json.loads((A/'INPUT_HASHES.json').read_text());assert len(inputs)==13000
scores=defaultdict(list);coverage=defaultdict(int);sets=defaultdict(int);diff=defaultdict(list);counts=defaultdict(int);checks=0;max_error=0.;start=time.monotonic()
def close(a,b):
 global checks,max_error
 error=abs(a-b);max_error=max(max_error,error);checks+=1
 assert error<=1e-9+2e-12*abs(b),(a,b,error)
for i,(name,h) in enumerate(inputs.items()):
 raw=(R/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h
 d=json.loads(gzip.decompress(raw));assert d['context_sha256']==s['context_sha256']
 for r in d['rows']:
  key=f"{r['mechanism']}_b{r['budget']}";counts[key]+=1
  for p in r['points']:
   error=p['estimate']-p['truth'];close(error,p['signed_error']);close(error*error,p['squared_error'])
   scores[(key,p['method'],'mse')].append(error*error);scores[(key,p['method'],'signed_error')].append(error)
  iv={v['interval_id']:v for v in r['intervals']+r['strong_intervals']}
  for n,v in iv.items():
   B=r['points'][0]['truth'];hit=v['status']=='OK' and v['lower']<=B<=v['upper'];assert hit==v['covers']
   width=r['capacity'] if v['status']=='EMPTY' else v['upper']-v['lower'];close(width,v['scored_width'])
   scores[(key,n,'mean_scored_width')].append(width);coverage[key,n]+=int(hit)
   if 'accepted_set_covers' in v:sets[key,n]+=int(v['accepted_set_covers'])
  if r['budget']==200 and r['mechanism'] in [f'M{x:02}' for x in range(1,7)]:diff[r['mechanism']].append(iv['rlfa_past_certified']['scored_width']-iv['seq_detector_100']['scored_width'])
 if (i+1)%3000==0:print(json.dumps({'reviewed_paths':i+1}),flush=True)
assert len(counts)==39 and set(counts.values())=={1000}
for (cell,method,metric),values in scores.items():
 obj=s['cells'][cell]['intervals' if metric=='mean_scored_width' else 'points'][method];close(math.fsum(values)/len(values),obj[metric]);se=math.sqrt(variance(values)/len(values));close(se,obj[{'mse':'mse_mcse','signed_error':'signed_error_mcse','mean_scored_width':'width_mcse'}[metric]])
def limits(k,n):
 if k==0:lo=0.
 else:
  l,h=0.,1.
  for _ in range(55):
   mid=(l+h)/2
   if binom.sf(k-1,n,mid)<.025:l=mid
   else:h=mid
  lo=(l+h)/2
 if k==n:hi=1.
 else:
  l,h=0.,1.
  for _ in range(55):
   mid=(l+h)/2
   if binom.cdf(k,n,mid)>.025:l=mid
   else:h=mid
  hi=(l+h)/2
 return lo,hi
for (cell,m),k in coverage.items():
 v=s['cells'][cell]['intervals'][m];assert k==v['hull_successes'];close(k/1000,v['hull_coverage'])
 for a,b in zip(limits(k,1000),v['hull_cp95']):close(a,b)
 if (cell,m) in sets:
  assert sets[cell,m]==v['set_successes']
  for a,b in zip(limits(sets[cell,m],1000),v['set_cp95']):close(a,b)
for c in s['primary_contrasts']:
 values=diff[c['cell']];mu=mean(values);v=variance(values);n=len(values);se=math.sqrt(v/n)
 close(mu,c['difference']);close(se,c['mcse']);rad=math.sqrt(2*v*math.log(240)/n)+14*20000*math.log(240)/(3*(n-1))
 close(max(-20000,mu-rad),c['simultaneous_finite95'][0]);close(min(20000,mu+rad),c['simultaneous_finite95'][1])
fam=sorted(s['coverage_family'],key=lambda x:x['p_undercoverage']);last=0
for i,f in enumerate(fam):
 key=f"{f['cell']}_b{f['budget']}";k=sets[key,f['interval']];p=float(binom.cdf(k,1000,.95));close(p,f['p_undercoverage']);last=max(last,min(1,(117-i)*p));close(last,f['holm_p']);assert f['holm_reject']==(last<=.05)
receipt={'status':'PASS','paths':len(inputs),'cells':len(counts),'arithmetic_comparisons':checks,'maximum_absolute_difference':max_error,'seconds':time.monotonic()-start,'scope':'Separate raw-record mean/variance/coverage reduction, exact-binomial tail inversion and Holm calculation. Not a second interval implementation or independent measurement replication.','summary_sha256':hashlib.sha256((A/'SUMMARY.json').read_bytes()).hexdigest(),'review_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(A/'INDEPENDENT_ARITHMETIC_REVIEW.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
