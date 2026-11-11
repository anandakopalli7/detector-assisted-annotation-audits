"""Optional bounded local replay of one predeclared path; never rents compute."""
from pathlib import Path
import argparse,hashlib,json,multiprocessing as mp,os,sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parent
def evaluate(cell,replicate,path,context):
 sys.path.insert(0,str(ROOT/'04_code'))
 from ojcs_engine import synthetic,evaluate_pair
 from ojcs_runner import external_frames,validate,store_checkpoint
 if cell.startswith('COCO_'):pub,y,frame=external_frames()[cell]
 else:pub,y=synthetic(cell,replicate);frame=None
 rows=evaluate_pair(pub,y,cell,replicate,budgets=(50,100,200),frame_id=frame)
 validate(rows,(cell,replicate))
 store_checkpoint(path,context,(cell,replicate),rows)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cell',required=True,choices=[f'M{i:02}' for i in range(1,11)]+['COCO_uniform','COCO_height_dependent','COCO_rare'])
 p.add_argument('--replicate',required=True,type=int)
 p.add_argument('--execute',action='store_true',help='Explicitly run local computation; otherwise only verify and describe the path')
 p.add_argument('--max-seconds',type=int,default=600)
 a=p.parse_args();assert 0<=a.replicate<1000 and 1<=a.max_seconds<=3600
 from verify_artifacts import check
 check()
 context=hashlib.sha256((ROOT/'CODE_MANIFEST.json').read_bytes()).hexdigest()
 out=ROOT/'05_results/LOCAL_REPRODUCTION'/f'{a.cell}_{a.replicate:04}.json.gz'
 assert not out.exists() and not out.with_suffix(out.suffix+'.sha256').exists(),'Existing replay output is preserved'
 if not a.execute:
  print(json.dumps({'status':'PLAN_ONLY','cell':a.cell,'replicate':a.replicate,'budgets':[50,100,200],'maximum_seconds':a.max_seconds,'output':str(out),'network_or_cloud_actions':False}));return
 out.parent.mkdir(parents=True,exist_ok=True)
 child=mp.get_context('spawn').Process(target=evaluate,args=(a.cell,a.replicate,str(out),context));child.start();child.join(a.max_seconds)
 if child.is_alive():
  child.terminate();child.join(5)
  if child.is_alive():child.kill();child.join(5)
  raise TimeoutError('Local replay deadline reached; partial evidence, if any, is preserved')
 if child.exitcode!=0 or not out.is_file():raise RuntimeError('Local replay failed; original evidence unchanged')
 print(json.dumps({'status':'COMPLETE','output':str(out),'note':'Scientific kernels and seeds are unchanged. Release context and timing differ from the original execution.'}))
if __name__=='__main__':main()
