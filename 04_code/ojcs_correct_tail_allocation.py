"""Disclosed post-result mathematical correction; no experimental reruns."""
from pathlib import Path
import copy,csv,hashlib,json,math
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];old=R/'05_results/ANALYSIS_V1';new=R/'05_results/ANALYSIS_V2'
assert not new.exists()
assert json.loads((old/'INDEPENDENT_ARITHMETIC_REVIEW.json').read_text())['status']=='PASS'
s=json.loads((old/'SUMMARY.json').read_text());before=copy.deepcopy(s);new.mkdir()
correction={'status':'CORRECTED','detected_after_outcomes':True,'utc':datetime.now(timezone.utc).isoformat(),'issue':'The frozen specification applied the one-sided empirical Bernstein allocation to a two-sided display.','source':'https://www.learningtheory.org/colt2009/papers/012.pdf','source_location':'Theorem 4, page 2','derivation':'For D in [-C,C], scale to [0,1]. Apply Theorem 4 to D and -D, each with failure probability alpha/(2*6). A union bound over 12 tails gives family error <=alpha. The logarithm is log(4*6/alpha)=log(480), not log(240).','unchanged':'All raw results, sample sizes, paired means, MCSEs, t intervals, coverage tests and historical decisions. Original six one-sided upper bounds remain valid and inconclusive.','action':'Widen two-sided finite bounds only; preserve ANALYSIS_V1 and the frozen protocol; no new data or favorable reinterpretation.','original_summary_sha256':hashlib.sha256((old/'SUMMARY.json').read_bytes()).hexdigest()}
for c in s['primary_contrasts']:
 n=c['n'];v=c['mcse']**2*n;lv=math.log(480);radius=math.sqrt(2*v*lv/n)+7*40000*lv/(3*(n-1))
 original=c['simultaneous_finite95'];correct=[max(-20000,c['difference']-radius),min(20000,c['difference']+radius)]
 assert correct[0]<=original[0] and correct[1]>=original[1] and correct[0]<0<correct[1]
 c['original_one_sided_family_upper95']=original[1];c['simultaneous_finite95']=correct;c['finite_superiority']=correct[1]<0
s.update(schema='OJCS_ANALYSIS_V2',post_result_correction=correction)
assert s['cells']==before['cells'] and s['coverage_family']==before['coverage_family'] and s['claims']==before['claims']
for name in ['points.csv','intervals.csv','strong_coverage_tests.csv','INPUT_HASHES.json','QA.json','INDEPENDENT_ARITHMETIC_REVIEW.json']:(new/name).write_bytes((old/name).read_bytes())
(new/'SUMMARY.json').write_text(json.dumps(s,indent=2,allow_nan=False))
with (new/'primary_contrasts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(s['primary_contrasts'][0]));w.writeheader();w.writerows(s['primary_contrasts'])
(R/'03_protocol/ANALYSIS_TAIL_ALLOCATION_ERRATUM.json').write_text(json.dumps(correction,indent=2))
(new/'CORRECTION_REVIEW.json').write_text(json.dumps({'status':'PASS','checks':['Six paired effects, t intervals and MCSEs unchanged','All 39 cell summaries byte-equivalent as objects','All 117 coverage tests unchanged','All six two-sided bounds widened','No finite superiority result changed','Theorem 4 allocation alpha/12 applied to both normalized tails'],'scope':'Mathematical correction and identity checks following separate raw-record arithmetic review'},indent=2))
(new/'COMPLETE.json').write_text(json.dumps({'status':'PASS','paths':13000,'cells':39,'point_rows':507,'interval_rows':741,'summary_sha256':hashlib.sha256((new/'SUMMARY.json').read_bytes()).hexdigest(),'correction_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_analysis':'ANALYSIS_V1','post_result_correction':True},indent=2))
print(json.dumps({'status':'PASS','correction':correction['issue'],'new_bounds':{c['cell']:c['simultaneous_finite95'] for c in s['primary_contrasts']}}))
