"""Known-answer fixtures and timing; never generate the proposed populations."""
from itertools import permutations,combinations
from pathlib import Path
from unittest.mock import patch
import copy,hashlib,json,math,os,subprocess,sys,time
import shutil
import numpy as np
from scipy.stats import beta,t
from paper2_generators_v01 import PublicFrame,ConstructionTruth
from paper2_evaluation_v02 import evaluate
from paper2_stratified_tie_correction_v01 import evaluate_stratified
from .methods import Frame,Measurements,METHODS,LEGACY,run_method,fold_ids,power_part,ActiveCS,active
from .engine import paired,streams
from .review import verify_row,summarize,cp,mc_interval
from .storage import canonical,file_sha,load,new_json,write_chunk,read_chunk,WorkerLock,owned,execution_permission
from .trace_review import replay_row

def make_fixture(N=20,K=20,kind='varied'):
    i=np.arange(N);q=i%3;original=(i*3)%7
    y=((i*7+3)%(K+1)).astype(int)
    if kind=='zero':q=np.zeros(N,dtype=int);y=np.zeros(N,dtype=int)
    if kind=='max':y=np.full(N,K,dtype=int)
    if kind=='constant':q=np.ones(N,dtype=int)
    # All arithmetic choices are artificial test inputs, not research models.
    drive=i//10 if N%10==0 else np.zeros(N,dtype=int)
    features=np.column_stack((q,original,original+q,q,np.zeros(N),np.zeros(N))).astype(float)
    public=PublicFrame(features,original,original+q,q,original,drive,np.full(N,K),np.array([]),np.array([]),np.array([]))
    truth=ConstructionTruth(y,y.copy(),y.copy(),y.copy())
    return public,truth,Frame.from_public(public,'FABRICATED_'+str(N)+'_'+kind)

def fixture_inputs(N,b):
    # Different fixed streams; no prospective namespace or generator is used.
    return (np.random.Generator(np.random.PCG64(1701)).permutation(N),
            [np.random.Generator(np.random.PCG64(1702+k)).permutation(N//2) for k in (0,1)],
            np.random.Generator(np.random.PCG64(1704)).random(b))

def fixture_checks(output,root):
    output=owned(output,root);output.mkdir(parents=True,exist_ok=False);checks=0;groups=[]
    def check(x,msg):
        nonlocal checks
        checks+=1
        if not x:raise AssertionError(msg)
    def raises(fn,types=(Exception,)):
        try:fn()
        except types:check(True,'raised');return
        raise AssertionError('expected rejection')
    try:
        for kind in ('varied','zero','max','constant'):
            public,truth,frame=make_fixture(kind=kind)
            for budget in (10,20):
                order,fo,u=fixture_inputs(frame.N,budget)
                for method in METHODS:
                    out=run_method(frame,budget,method,lambda i:truth.omitted[i],order,fo,u)
                    check(out['labels']==budget and len(set(out['trace']['measurement_ids']))==budget,'budget')
                    if method in LEGACY:
                        ref=evaluate_stratified(public,truth,order,budget) if method=='stratified_HT' else evaluate(public,truth,order,budget,method)
                        check(math.isclose(out['estimate'],ref['estimate'],rel_tol=1e-12,abs_tol=1e-10),'legacy point')
                        ci=out['intervals'][method]
                        check((ci['lower'],ci['upper'],ci['status'])==(ref['lower'],ref['upper'],ref['status']),'legacy interval')
                    if budget==frame.N:
                        check(abs(out['estimate']-sum(truth.omitted))<1e-9,'census point')
                        for ci in out['intervals'].values():check(ci['lower']==ci['upper']==sum(truth.omitted),'census interval')
                    altered=truth.omitted.copy();unseen=set(range(frame.N))-set(out['trace']['measurement_ids'])
                    for i in unseen:altered[i]=(altered[i]+7)%(frame.K+1)
                    other=run_method(frame,budget,method,lambda i:altered[i],order,fo,u)
                    check(out['estimate']==other['estimate'] and out['trace']==other['trace'] and out['intervals']==other['intervals'],'future measurement leakage')
                    if method.startswith('frame_cf2_'):
                        for rec in out['trace']['folds']:check(not set(rec['training_ids'])&set(rec['fold_ids']),'own-fold training leak')
        groups.append('legacy agreement, census, budgets, unseen-label mutation, fold separation')

        # Independent hand example for the real fitted-weight implementation.
        y=np.array([0.,0.,1.]);q=np.array([0.,1.,2.]);est=[]
        for s in combinations(range(3),2):
            ids=np.array(s);v,_,_=power_part((q,q[ids]),y[ids],3);est.append(v)
        check(est==[0.,1.5,0.] and np.mean(est)==.5,'power counterexample')
        groups.append('fitted-weight exact hand counterexample')

        # Enumeration of REAL active implementation on small fixed arrays.
        public,truth,frame=make_fixture(4,2);budget=2
        for lam in (0.,.5,1.):
            total_prob=expected=wealth=0.
            for order in permutations(range(4),budget):
                obs=Measurements(lambda i:truth.omitted[i],4,2)
                point,ci,tr=active(frame,obs,budget,np.zeros(budget),lam,forced_order=order)
                pr=math.prod(d['p'] for d in tr['draws']);total_prob+=pr;expected+=pr*point
                c=frame.q-frame.q.mean();oracle=0.
                for j,i in enumerate(order):
                    prob=tr['draws'][j]['p'];v=1+(4-budget)/(4-j-1)*(1/((4-j)*prob)-1)
                    oracle+=4/budget*v*(truth.omitted[i]-lam*c[i])
                check(abs(oracle-point)<1e-12,'independent LURE arithmetic')
                cs=ActiveCS(tr['normalized_observations'],tr['conditional_mean_centers'],tr['conditional_mean_scales'],[truth.omitted[i] for i in order],4,2,alpha=.4)
                B=int(sum(truth.omitted));null=(B-cs.center)/cs.scale
                factors=1+np.minimum(cs.bets,.99/np.maximum(null,1e-300))*(cs.x-null)
                check(np.all(factors>=0),'nonnegative factors');wealth+=pr*float(np.prod(factors))
                threshold=math.log(2/cs.alpha)+1e-10
                grid=list(range(cs.lo,cs.hi+1));pos=[v for v in grid if cs.max_log_wealth(v)[0]<=threshold];neg=[v for v in grid if cs.max_log_wealth(v)[1]<=threshold]
                got=cs.bounds()
                if not pos or not neg:check(got.status=='EMPTY','empty inversion')
                else:check((got.lower,got.upper)==(max(cs.lo,min(pos)-1),min(cs.hi,max(neg)+1)),'full grid vs bisection')
            check(abs(total_prob-1)<1e-12 and abs(expected-sum(truth.omitted))<1e-10,'active expectation')
            check(abs(wealth-1)<1e-10,'true-null expected wealth')
        groups.append('actual active paths, LURE oracle, unbiasedness, betting wealth, exhaustive inversion')

        public,truth,frame=make_fixture();order,fo,u=fixture_inputs(20,10)
        uniform=lambda mu,c,remaining,K,lam:(np.full(len(remaining),1/len(remaining)),0.)
        with patch('paper2_compare.methods.acquisition',uniform):
            obs=Measurements(lambda i:truth.omitted[i],20,20)
            point,ci,tr=active(frame,obs,10,u,1.,forced_order=order[:10])
        ref=run_method(frame,10,'raw_difference',lambda i:truth.omitted[i],order,fo,u)
        check(abs(point-ref['estimate'])<1e-10,'uniform raw reduction')
        check(ci==ref['intervals']['raw_difference'],'uniform interval reduction')
        raises(lambda:run_method(frame,10,'direct_HT',lambda i:21,order,fo,u),(ValueError,))
        raises(lambda:run_method(frame,10,'direct_HT',lambda i:0,order*0,fo,u),(ValueError,))
        raises(lambda:run_method(frame,1,'ff_power',lambda i:0,order,fo,u),(ValueError,))
        raises(lambda:run_method(frame,9,'frame_cf2_offset100',lambda i:0,order,fo,u),(ValueError,))
        groups.append('uniform-design reduction and invalid-input rejection')
        # Tiny budgets remain useful algorithm edge cases even though research
        # budgets start at 50. A b=1 Wald interval is explicitly undefined.
        for b in (1,2):
            for method in ('seq_detector_100','active_count_lure','ppat_count_half','ppat_count_one'):
                small=run_method(frame,b,method,lambda i:truth.omitted[i],order,fo,u)
                check(small['labels']==b,'tiny active/sequential budget')
        from dataclasses import replace
        sparse=np.zeros(20,dtype=int);sparse[0]=1
        tiny=replace(public,unmatched=sparse,features=np.column_stack((sparse,np.zeros((20,5)))))
        tf=Frame.from_public(tiny,'FABRICATED_TINY_STRATUM')
        for method in ('stratified_HT','ff_strat_power'):
            o=run_method(tf,4,method,lambda i:truth.omitted[i],order,fo,u)
            check(o['trace']['allocation']==[3,1,0] and o['labels']==4,'one-unit census stratum')
        raises(lambda:run_method(frame,3,'ff_strat_power',lambda i:truth.omitted[i],order,fo,u),(ValueError,))
        groups.append('tiny budgets, tiny census stratum, undefined variance rejection')

        cfg=load(Path(__file__).resolve().parents[1]/'design_config.json')
        fc=copy.deepcopy(cfg);fc.update(N=20,K=20,mechanisms=['M01'],primary_mechanisms=['M01'],budgets=[10],primary_budget=10,populations_per_mechanism=1,replicates_per_cell=2,chunk_size=2)
        fc['expected']={'paired_chunks':1,'paired_replicates':2,'point_method_records':26,'interval_records':32,'point_cells':13,'interval_cells':16}
        rows=[paired(frame,truth.omitted,10,order,fo,u,{'mechanism':'M01','budget':10,'replicate':r,'population_index':0}) for r in range(2)]
        for row in rows:verify_row(row,fc);check(True,'row validates')
        for row in rows:replay_row(row,frame,truth.omitted,order,fo,u);check(True,'full sampling and point replay')
        for mutation in ('vector_hash','uniform_choice','fold_stream','label_value','point'):
            row=copy.deepcopy(rows[0])
            if mutation=='vector_hash':row['traces']['ppat_count_half']['draws'][0]['probabilities_sha256']='0'*64
            elif mutation=='uniform_choice':row['traces']['active_count_lure']['draws'][0]['p']*=1.0001
            elif mutation=='fold_stream':row['traces']['frame_cf2_offset100']['folds'][0]['sample_ids'].reverse()
            elif mutation=='label_value':row['traces']['ff_power']['measurement_values'][0]+=1
            else:row['points'][0]['estimate']+=1
            raises(lambda rr=row:replay_row(rr,frame,truth.omitted,order,fo,u),(ValueError,))
        for corruption in ('missing_method','bad_square','bad_width','wrong_label_count','bad_propensity','leakage'):
            row=copy.deepcopy(rows[0])
            if corruption=='missing_method':row['points'].pop()
            elif corruption=='bad_square':row['points'][0]['squared_error']+=1
            elif corruption=='bad_width':row['intervals'][0]['scored_width']+=1
            elif corruption=='wrong_label_count':row['traces']['direct_HT']['measurement_ids'].pop()
            elif corruption=='bad_propensity':row['traces']['active_count_lure']['draws'][0]['p']=0
            else:row['traces']['frame_cf2_offset100']['folds'][0]['training_ids'].append(row['traces']['frame_cf2_offset100']['folds'][0]['fold_ids'][0])
            raises(lambda rr=row:verify_row(rr,fc),(ValueError,))
        run=output/'fixture_run';(run/'chunks').mkdir(parents=True);context='f'*64
        chunk=run/'chunks'/'M01_b10_r00000.json.gz';write_chunk(chunk,{'context_sha256':context,'rows':rows})
        check(read_chunk(chunk,context)['rows']==rows,'checkpoint roundtrip')
        raises(lambda:write_chunk(chunk,{'context_sha256':context,'rows':rows}),(FileExistsError,))
        raises(lambda:read_chunk(chunk,'a'*64),(ValueError,))
        partial=run/'chunks'/'M01_b10_r00002.json.gz.partial';partial.write_bytes(b'preserved interruption')
        from .engine import ensure_clean_partials
        raises(lambda:ensure_clean_partials(run),(RuntimeError,))
        # Summary reader ignores partials; execution refuses recovery. Fixture
        # verifies the intact checkpoint arithmetic without removing the partial.
        summary=summarize(run,context,fc,strict=True)
        check(summary['counts']==fc['expected'],'summary counts')
        for item in summary['points']:
            raw=next(p for p in rows[0]['points'] if p['method']==item['method'])
            check(item['mse']==raw['squared_error'],'independent mean MSE')
        check(len(summary['primary_effects'])==24,'24 effect outputs')
        a=np.array([-.2,.1,.4,.5]);result=mc_interval(a,2)
        check(abs(result['mc_standard_error']-np.std(a,ddof=1)/2)<1e-14,'MC SE')
        check(cp(0,10,.05)[0]==0 and cp(10,10,.05)[1]==1,'CP boundary')
        L=math.log(4*24/.05);rad=math.sqrt(2*float(np.var(a,ddof=1))*L/4)+7*2*L/9
        check(abs(result['bounded_empirical_bernstein_interval'][1]-(float(a.mean())+rad))<1e-12,'independent MC bound')
        wrong=copy.deepcopy(fc);wrong['replicates_per_cell']=3
        raises(lambda:summarize(run,context,wrong,strict=True),(ValueError,))
        raises(lambda:owned(Path(root).parent/'forbidden_output',root),(ValueError,))
        with WorkerLock(output/'lock_test'):
            child=subprocess.run([sys.executable,'-B','-c','from paper2_compare.storage import WorkerLock; from pathlib import Path;\nwith WorkerLock(Path(__import__("sys").argv[1])): pass',str(output/'lock_test')],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
            check(child.returncode!=0 and 'another worker' in child.stderr,'duplicate worker rejected')
        with WorkerLock(output/'lock_test'):check(True,'lock released')
        no_approval=output/'NOT_EXECUTION_APPROVAL.json';new_json(no_approval,{'decision':'D-P2-12','status':'APPROVED_IMPLEMENTATION_ONLY'})
        raises(lambda:execution_permission(output/'no_manifest',no_approval,output/'never_created',root),(PermissionError,))
        check(not (output/'never_created').exists(),'consent before writes/generation')
        # Exercise the real controller with an explicit fixture-only dependency
        # injection, never a forged execution receipt or the research generator.
        from . import engine
        fake_manifest=output/'FIXTURE_MANIFEST_ONLY.json';new_json(fake_manifest,{'scope':'FABRICATED_CONTROLLER_TEST'})
        fixture_context=file_sha(fake_manifest);fc['expected']['populations']=1
        fc['population_seed_prefix']='FABRICATED_CONTROLLER_POPULATION';fc['audit_seed_prefix']='FABRICATED_CONTROLLER_AUDIT'
        def factory(mech,index,cfg,directory,ctx):
            path=Path(directory)/f'{mech}_{index:03}.npz'
            if not path.exists():
                with path.open('xb') as fh:np.savez_compressed(fh,**public.__dict__,omitted=truth.omitted)
                new_json(path.with_suffix('.json'),{'sha256':file_sha(path),'context_sha256':ctx,'mechanism':mech,'index':index,'seed':engine.seed(f"{cfg['population_seed_prefix']}|{mech}|{index}")})
            return engine.load_population(mech,index,cfg,directory,ctx)
        full=output/'controller_fixture';receipt=no_approval
        def controller(target):
            with patch('paper2_compare.engine.execution_permission',return_value=(target,{},fc)),patch('paper2_compare.engine.construct_population',factory):
                engine.run(fake_manifest,receipt,target,root)
        controller(full);complete=load(full/'COMPLETE.json')
        check(complete['counts']['paired_replicates']==2,'controller complete')
        check(load(full/'FINAL_SUMMARY.json')==summarize(full,fixture_context,fc,True,True),'separate full reaggregation and replay')
        raises(lambda:controller(full),(RuntimeError,))
        resumed=output/'controller_resume_fixture';resumed.mkdir()
        shutil.copy2(full/'RUN_CONTEXT.json',resumed/'RUN_CONTEXT.json')
        shutil.copytree(full/'populations',resumed/'populations');shutil.copytree(full/'chunks',resumed/'chunks')
        hashes={p.name:file_sha(p) for p in (resumed/'chunks').iterdir()}
        controller(resumed)
        check(hashes=={p.name:file_sha(p) for p in (resumed/'chunks').iterdir()},'safe checkpoint resume unchanged')
        check((resumed/'COMPLETE.json').is_file(),'resume completion')
        groups.append('actual controller with fabricated bank, complete refusal, safe checkpoint resume, full trace replay')
        groups.append('raw-record corruption, immutable checkpoints, partial preservation, aggregation, lock and consent gates')
        report={'status':'PASS','scope':'KNOWN_ANSWER_SOFTWARE_FIXTURES_ONLY_NOT_RESEARCH_EFFICACY','assertions':checks,'groups':groups,'performance_evaluation_populations_generated':0}
        new_json(output/'FIXTURE_REPORT.json',report);return report
    except BaseException as e:
        new_json(output/'FIXTURE_FAILURE.json',{'type':type(e).__name__,'message':str(e),'passed_assertions_before_failure':checks,'scope':'implementation fixture; no performance study'})
        raise

def profile(output,root):
    output=owned(output,root);output.mkdir(parents=True,exist_ok=False)
    public,truth,frame=make_fixture(1000,20);timings=[];sizes=[];review_timings=[]
    # Two prescribed passes, all methods/all budgets, no outcome-based choice.
    for repeat in range(2):
        for b in (50,100,200,400):
            order,fo,u=fixture_inputs(1000,b);started=time.perf_counter()
            row=paired(frame,truth.omitted,b,order,fo,u,{'mechanism':'FABRICATED_PROFILE','budget':b,'replicate':repeat,'population_index':0})
            timings.extend({'repeat':repeat,'budget':b,'method':p['method'],'seconds':p['elapsed_seconds']} for p in row['points'])
            import gzip
            sizes.append({'repeat':repeat,'budget':b,'json_bytes':len(canonical(row)),'gzip_bytes':len(gzip.compress(canonical(row),mtime=0)),'paired_seconds':time.perf_counter()-started})
            start=time.perf_counter();replay_row(row,frame,truth.omitted,order,fo,u)
            review_timings.append({'repeat':repeat,'budget':b,'seconds':time.perf_counter()-start})
    total=sum(t['seconds'] for t in timings)/2*10*5000
    disk=sum(s['gzip_bytes'] for s in sizes)/2*10*5000
    report={'scope':'FABRICATED_RUNTIME_ONLY_NO_COMPARATIVE_EFFICACY','timings':timings,'sizes':sizes,
            'raw_execution_hours_extrapolation':total/3600,'estimate_range_hours':[total/3600*.7,total/3600*2],
            'compressed_record_GiB_extrapolation':disk/1024**3,
            'review_timings':review_timings,'trace_review_hours_extrapolation':sum(x['seconds'] for x in review_timings)/2*10*5000/3600,
            'paired_including_serialization_hours_extrapolation':sum(x['paired_seconds'] for x in sizes)/2*10*5000/3600,
            'limitations':['Two fixed passes on one fabricated frame, not prospective populations','Runtime varies with numerical inversion/OS load','Includes trace evidence; aggregation, I/O and verification add work','No scientific conclusion from these inputs']}
    new_json(output/'PROFILE_REPORT.json',report);return {k:v for k,v in report.items() if k not in ('timings','sizes')}
