"""Reduce the two completed fixed studies; never generates data or runs methods."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,sys,time
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
from scipy.stats import beta,binom,t
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def cp(k,n,alpha=.05):return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]
def holm(p):
 a=np.asarray(p,float);ix=np.argsort(a,kind='stable');out=np.empty(len(a));out[ix]=np.minimum(1,np.maximum.accumulate(a[ix]*(len(a)-np.arange(len(a)))));return out.tolist()
def meanse(a):
 a=np.asarray(a,float);return float(a.mean()),float(a.std(ddof=1)/math.sqrt(len(a)))
def width_contrast(a,b,capacity=20000):
 a=np.asarray(a,float);b=np.asarray(b,float);d=a-b;n=len(d);m,se=meanse(d);q=float(t.ppf(.975,n-1));lv=math.log(2/(.05/6))
 radius=math.sqrt(2*float(d.var(ddof=1))*lv/n)+7*(2*capacity)*lv/(3*(n-1))
 return dict(n=n,rlfa_mean=float(a.mean()),anchor_mean=float(b.mean()),difference=m,ratio=float(a.mean()/b.mean()),mcse=se,t95=[m-q*se,m+q*se],simultaneous_finite95=[max(-capacity,m-radius),min(capacity,m+radius)],finite_superiority=m+radius<0)
def qa():
 assert cp(0,10)[0]==0 and cp(10,10)[1]==1
 assert abs(cp(10,10)[0]-(.025**.1))<1e-13
 assert abs(cp(0,10)[1]-(1-.025**.1))<1e-13
 assert np.allclose(holm([.03,.001,.04]),[.06,.003,.06])
 x=width_contrast(np.arange(1000)+10,np.arange(1000)+20)
 assert x['difference']==-10 and x['mcse']==0 and x['t95']==[-10,-10] and not x['finite_superiority']
 assert abs(float(binom.cdf(0,10,.95))-.05**10)<1e-20
 assert np.allclose(meanse([1,2,3]),[2,1/math.sqrt(3)])
 return {'status':'PASS','checks':9,'scope':'analytic endpoints, known Holm example, paired constant contrast, mean/SE, exact binomial boundary'}
def table(path,rows):
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
 sys.path[:0]=[str(ROOT/'04_code'),str(ROOT/'04_code/deps')]
 from ojcs_runner import ids,checkpoint_path,load_checkpoint,validate,verify_sources
 out=ROOT/'05_results/ANALYSIS_V1'
 if out.exists():raise RuntimeError('Preserve existing analysis; no overwrite')
 manifest=ROOT/'03_protocol/EXECUTION_MANIFEST.json';context=sha(manifest);verify_sources(json.loads(manifest.read_text()))
 out.mkdir();write(out/'QA.json',qa());started=time.monotonic()
 groups={};digests=[];seen=0
 for phase,expected in [('synthetic',10000),('external',3000)]:
  d=ROOT/('05_results/STUDY_'+phase.upper());c=json.loads((d/'COMPLETE.json').read_text())
  assert c['context_sha256']==context and c['paths']==expected and not c['fixture'] and c['row_validation']=='PASS'
  assert not list(d.glob('FAILURE_*.json')) and not list(d.rglob('*.partial'))
  assert len(list((d/'paths').glob('*.json.gz')))==expected
  for job in ids(phase):
   p=checkpoint_path(d,job);rows=load_checkpoint(p,context,job);validate(rows,job);digests.append((str(p.relative_to(ROOT)),sha(p)))
   for r in rows:
    k=(r['mechanism'],r['budget']);g=groups.setdefault(k,{'points':defaultdict(list),'intervals':defaultdict(list),'risk':[],'truth':[],'capacity':r['capacity'],'replicates':[]})
    g['replicates'].append(r['replicate']);g['truth'].append(r['points'][0]['truth']);g['risk'].append(r['sequential_risk_score'])
    for point in r['points']:g['points'][point['method']].append(point)
    for ci in r['intervals']+r['strong_intervals']:
     ci=dict(ci)
     if 'full_logical_support' not in ci:
      tr=r['traces'][ci['method']];v=tr['measurement_values'];lo=sum(v);hi=lo+(1000-len(v))*20
      ci['full_logical_support']=ci['status']=='OK' and ci['lower']<=lo and ci['upper']>=hi
     g['intervals'][ci['interval_id']].append(ci)
   seen+=1
   if seen%1000==0:print(json.dumps({'validated_paths':seen,'total':13000}),flush=True)
 summary={'schema':'OJCS_ANALYSIS_V1','context_sha256':context,'analysis_spec_sha256':sha(ROOT/'03_protocol/ANALYSIS_SPEC.json'),'script_sha256':sha(__file__),'paths':seen,'rows':seen*3,'cells':{},'primary_contrasts':[],'coverage_family':[],'claims':{}}
 point_table=[];interval_table=[];family=summary['coverage_family']
 for (cell,b),g in sorted(groups.items()):
  assert sorted(g['replicates'])==list(range(1000)) and len(g['points'])==13 and len(g['intervals'])==19
  result={'n':1000,'unit':'independent generated population plus paired audit' if cell.startswith('M') else 'audit order on one fixed frame','truth_min':min(g['truth']),'truth_max':max(g['truth']),'points':{},'intervals':{}}
  for method,ps in g['points'].items():
   mse,mse_se=meanse([p['squared_error'] for p in ps]);bias,bias_se=meanse([p['signed_error'] for p in ps]);timing=[p['elapsed_seconds'] for p in ps]
   v={'mse':mse,'mse_mcse':mse_se,'signed_error':bias,'signed_error_mcse':bias_se,'mean_elapsed_seconds':float(np.mean(timing)),'median_elapsed_seconds':float(np.median(timing)),'labels_min':min(p['labels'] for p in ps),'labels_max':max(p['labels'] for p in ps)}
   assert v['labels_max']<=b;result['points'][method]=v;point_table.append(dict(cell=cell,budget=b,method=method,**v))
  risk,riskse=meanse(g['risk']);rd=[p['squared_error']-q for p,q in zip(g['points']['seq_detector_100'],g['risk'])];rm,rse=meanse(rd)
  result['risk']={'mean_oracle_risk':risk,'risk_mcse':riskse,'mean_squared_error_minus_risk':rm,'paired_mcse':rse,'interpretation':'oracle predictable risk; not observed conditional MSE'}
  for name,cs in g['intervals'].items():
   n=len(cs);k=sum(c['covers'] for c in cs);width,wse=meanse([c['scored_width'] for c in cs]);empty=sum(c['status']=='EMPTY' for c in cs)
   v={'mean_scored_width':width,'width_mcse':wse,'hull_successes':k,'hull_coverage':k/n,'hull_cp95':cp(k,n),'empty_count':empty,'full_logical_support_count':sum(c['full_logical_support'] for c in cs),'guarantee':cs[0]['guarantee']}
   if 'accepted_set_covers' in cs[0]:
    kset=sum(c['accepted_set_covers'] for c in cs)
    v.update(set_successes=kset,set_coverage=kset/n,set_cp95=cp(kset,n),mean_elapsed_seconds=float(np.mean([c['elapsed_seconds'] for c in cs])))
    family.append(dict(cell=cell,budget=b,interval=name,successes=kset,n=n,p_undercoverage=float(binom.cdf(kset,n,.95))))
   result['intervals'][name]=v
   interval_table.append(dict(cell=cell,budget=b,interval=name,**{key:json.dumps(value) if isinstance(value,list) else value for key,value in v.items()},**({} if 'set_successes' in v else {'set_successes':None,'set_coverage':None,'set_cp95':None,'mean_elapsed_seconds':None})))
  summary['cells'][f'{cell}_b{b}']=result
  if cell in [f'M{i:02}' for i in range(1,7)] and b==200:
   summary['primary_contrasts'].append(dict(cell=cell,**width_contrast([x['scored_width'] for x in g['intervals']['rlfa_past_certified']],[x['scored_width'] for x in g['intervals']['seq_detector_100']])))
 assert len(family)==117
 for rec,adj in zip(family,holm([f['p_undercoverage'] for f in family])):rec.update(holm_p=adj,holm_reject=adj<=.05)
 summary['claims']={'finite_width_superiority_cells':[r['cell'] for r in summary['primary_contrasts'] if r['finite_superiority']],'strong_undercoverage_rejections':[r for r in family if r['holm_reject']],'historical_joint_target':'NOT_ATTAINED','new_experiments_complete':True,'submission_ready':False}
 write(out/'SUMMARY.json',summary);table(out/'points.csv',point_table);table(out/'intervals.csv',interval_table);table(out/'primary_contrasts.csv',summary['primary_contrasts']);table(out/'strong_coverage_tests.csv',family)
 write(out/'INPUT_HASHES.json',dict(digests));write(out/'COMPLETE.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','paths':seen,'cells':len(groups),'point_rows':len(point_table),'interval_rows':len(interval_table),'seconds':time.monotonic()-started,'summary_sha256':sha(out/'SUMMARY.json'),'input_ledger_sha256':sha(out/'INPUT_HASHES.json'),'analysis_sha256':sha(__file__)})
 print(json.dumps({'status':'COMPLETE','claims':summary['claims'],'primary_contrasts':summary['primary_contrasts']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--qa',action='store_true');a=p.parse_args()
 if a.qa:print(json.dumps(qa()))
 else:run()
