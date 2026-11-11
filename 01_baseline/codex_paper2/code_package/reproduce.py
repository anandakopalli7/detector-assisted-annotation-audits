"""Portable read-only checks. No research performance execution."""
from pathlib import Path
import sys,json,hashlib,math,itertools
P=Path(__file__).resolve().parent
def close(x,y):assert math.isclose(x,y,rel_tol=2e-13,abs_tol=1e-9),(x,y)
def verify():
    provenance=json.loads((P/'PROVENANCE.json').read_text())
    for n,v in provenance.items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==v['sha256'],n
    j=json.loads((P/'data/STORED_RISK_DIAGNOSTIC.json').read_text())
    assert j['new_audits']==0 and j['paired_audits']==120000 and j['method_records']==720000 and j['source_chunks']==1200
    for m,p in j['pooled'].items():
        cells=[x[m] for x in j['per_cell_sufficient_sums'].values()];n=sum(x['n'] for x in cells);assert n==120000
        for key,total in [('mean_q','sum_q'),('mean_sqrt_q','sum_sqrt_q'),('mean_recorded_width','sum_width'),('realized_mse','sum_se')]:close(p[key],sum(c[total] for c in cells)/n)
        close(p['normal_reference_width_mean_path'],3.92*p['mean_sqrt_q'])
        close(p['normal_reference_width_sqrt_mean_risk'],3.92*math.sqrt(p['mean_q']))
        assert p['normal_reference_width_mean_path']<=p['normal_reference_width_sqrt_mean_risk']+1e-9
        direct=j['pooled']['direct_HT']
        close(p['width_ratio_to_direct'],p['mean_recorded_width']/direct['mean_recorded_width'])
        close(p['mean_path_reference_ratio_to_direct'],p['mean_sqrt_q']/direct['mean_sqrt_q'])
        close(p['sqrt_mean_risk_ratio_to_direct'],math.sqrt(p['mean_q']/direct['mean_q']))
    e=json.loads((P/'data/DETECTOR_MATCHING_DIAGNOSTIC.json').read_text());assert len(e['rows'])==9
    for row in e['rows']:
        c=e['constructions'][row['case']];assert c['known_target']==c['sum_deleted_ids'] and c['cap_truncated']==0
        assert row['n11']+row['n10']+c['known_target']==1716
        line=f"{row['case']} & {row['threshold']:.1f} & {row['n11']} & {row['n10']} & {row['n01']} & {row['matched_outside_target']} & {row['matched_unresolved_mask']}"
        assert line in (P/'printed_tables/detector_matching.tex').read_text()
    close(e['n01_binomial_reference']['z'],(298-1716*.2)/math.sqrt(1716*.2*.8))
    for m,label in zip(['direct_HT','stratified_HT','raw_difference','split_ridge10','split_offset100','seq_detector_100'],['Direct HT','Stratified HT','Raw difference','Split ridge','Split offset','Sequential']):
        d=j['pooled'][m]
        assert f"{label} & {d['mean_recorded_width']:.2f} & {d['normal_reference_width_mean_path']:.2f} & {d['normal_reference_width_sqrt_mean_risk']:.2f}" in (P/'printed_tables/stored_risk_diagnostic.tex').read_text()
    from verify_additions import verify as verify_new
    extra=verify_new()
    return {'additional_checks':extra,'status':'PASS','copied_files':len(provenance),'risk_cells_checked':36,'matching_rows_checked':9,'new_experiments':0}
def smoke():
    sys.path.insert(0,str(P/'comparator'))
    from paper2_compare.fixtures import make_fixture,fixture_inputs
    from paper2_compare.methods import METHODS,run_method
    assertions=0
    for kind in ['varied','zero','max','constant']:
        public,truth,frame=make_fixture(kind=kind)
        for b in [10,20]:
            order,fo,u=fixture_inputs(20,b)
            for method in METHODS:
                o=run_method(frame,b,method,lambda i:truth.omitted[i],order,fo,u)
                assert o['labels']==b and len(set(o['trace']['measurement_ids']))==b;assertions+=1
                assert math.isfinite(o['estimate']);assertions+=1
                if method=='direct_HT':close(o['estimate'],20/b*sum(truth.omitted[order[:b]]));assertions+=1
                if b==20:close(o['estimate'],sum(truth.omitted));assertions+=1
    sys.path.insert(0,str(P/'integer'))
    from band import prefix,factor_pair,BandAudit
    from fractions import Fraction as F
    caps=(2,2,2);centers=(0,1,2);M=3;P0=sum(centers)
    for S in range(7):
        scores=[sum(F(1) if x==c else F(1,4) for x,c in zip(v,centers))/3 for v in itertools.product(range(3),repeat=3) if sum(v)==S]
        num,den=factor_pair(M,P0,prefix(centers),prefix(k-c for k,c in zip(caps,centers)),S,True)
        assert F(den,num)==max(scores);assertions+=1
    for relaxed in [False,True]:
        b=BandAudit(caps,relaxed)
        for i,v in enumerate((0,1,2)):b.prepare([centers[k] for k in b.remaining]);b.observe(i,v)
        assert b.accepted()==(3,);assertions+=1
    return {'status':'PASS','scope':'FABRICATED_CORRECTNESS_ONLY','assertions':assertions,'point_methods':len(METHODS),'new_performance_populations':0}
if __name__=='__main__':
    command=sys.argv[1] if len(sys.argv)>1 else 'verify'
    if command not in ('verify','smoke'):raise SystemExit('Choose verify or smoke; no experiment command is provided.')
    print(json.dumps(verify() if command=='verify' else smoke(),indent=2))
