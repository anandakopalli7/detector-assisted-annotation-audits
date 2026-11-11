"""Read-only, independent arithmetic for v03 descriptive additions and decisions."""
from pathlib import Path
import json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/'data'/n).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-9),(a,b)
def verify():
 s=read('DECISION_SUFFICIENT_STATS.json');h=read('FINAL_SYNTHETIC_REVIEW_v02.json');base=read('EXACT_BASELINE_INPUTS.json')
 baseline={m:math.fsum(r[m] for x in base.values() for r in x['population_risks'])/600 for m in ['direct_HT','stratified_HT']}
 for m,v in baseline.items():close(v,h['primary_mean_risk'][m])
 def upper(d):
  n=d['n'];mean=d['sum']/n;var=d['sum_centered_squares']/(n-1);z=math.log(2/d['delta'])
  close(mean,d['mean']);close(var,d['variance'])
  ans=mean+math.sqrt(2*var*z/n)+7*d['support_span']*z/(3*(n-1));close(ans,d['upper']);return ans
 riskupper=upper(s['risk']);decisions={}
 for m in baseline:
  decisions['Risk vs. '+('direct' if m=='direct_HT' else 'stratified')]=(riskupper*400000000-.9*baseline[m])
  close(decisions['Risk vs. '+('direct' if m=='direct_HT' else 'stratified')],h['risk_looks'][-1]['upper_contrasts'][m]*400000000)
  w=upper(s['width'][m])*20000
  close(w,h['width_results'][m]['upper']*20000)
  decisions['Width vs. '+('direct' if m=='direct_HT' else 'stratified')]=w
 for label,v in decisions.items():
  assert f'{label} & {v:,.3f} & '+('Yes' if v<=0 else 'No') in (P/'printed_tables/decisions.tex').read_text()
 assert sum(v<=0 for v in decisions.values())==2
 close(s['risk']['mean']*400000000,h['primary_mean_risk']['seq_detector_100'])
 # Independent inversion of Pr(Bin(n,p) >= k) at delta, via the small failure tail.
 c=min((x for x in h['cells'] if x['method']=='seq_detector_100'),key=lambda x:x['certificate_lower'])
 n=c['n'];k=c['certificate_count'];assert(n,k)==(20000,19933)
 def upper_tail(p):
  # Sum probability of no more than n-k failures using log probabilities.
  logs=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+(n-j)*math.log(p)+j*math.log1p(-p) for j in range(n-k+1)]
  pivot=max(logs);return math.exp(pivot)*math.fsum(math.exp(v-pivot) for v in logs)
 lo,hi=.9,1-1e-12
 for _ in range(80):
  mid=(lo+hi)/2
  if upper_tail(mid)>.05/48:hi=mid
  else:lo=mid
 close((lo+hi)/2,c['certificate_lower'])
 e=read('DETECTOR_MATCHING_DIAGNOSTIC.json');intact=read('DETECTOR_INTACT_SENSITIVITY.json')
 for r in e['rows']:
  assert r['n11']+r['n01']+r['matched_outside_target']==1840
  assert r['matched_unresolved_mask']<=r['matched_outside_target']
 for r in intact['rows']:
  assert r['proposals']==1840 and r['eligible_total']==1716
  assert r['eligible_matched_intact']+r['matched_outside_target']+r['unmatched_intact']==1840
  close(r['intact_sensitivity'],r['eligible_matched_intact']/1716)
  for case,c in r['deleted'].items():
   assert c['total']==e['constructions'][case]['known_target'];close(c['intact_sensitivity'],c['matched_intact']/c['total'])
  vals=' & '.join(f"{r['deleted'][c]['matched_intact']}/{r['deleted'][c]['total']}" for c in ['N01','N02','N03'])
  assert f"{r['threshold']:.1f} & {r['eligible_matched_intact']}/1716 & {r['unmatched_intact']} & {vals}" in (P/'printed_tables/intact_sensitivity.tex').read_text()
 r=read('RANGE_RECONSTRUCTION.json');assert len(r['rows'])==600
 assert {(x['mechanism'],x['replicate']) for x in r['rows']}=={(f'M{k:02d}',i) for k in range(1,7) for i in range(100)}
 assert all(x['draws']==200 and x['estimate_matches'] and x['risk_matches'] and 1<=x['min_d_over_K']<=x['max_d_over_K']<=2 for x in r['rows'])
 close(r['mean_d_over_K'],math.fsum(x['sum_d_over_K'] for x in r['rows'])/120000)
 return {'status':'PASS','decision_bounds':decisions,'baseline_populations':600,'limiting_certificate_count':k,'limiting_certificate_lower':(lo+hi)/2,'historical_paths_reconstructed':600,'new_performance_experiments':0,'proposals':1840,'scope':'Independent arithmetic from included aggregates and sufficient statistics. Does not redo detector inference or matching geometry, establish physical truth, or independently replicate the measurements.'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
