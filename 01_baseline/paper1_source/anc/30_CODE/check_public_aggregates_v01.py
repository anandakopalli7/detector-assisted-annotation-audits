"""Portable, read-only arithmetic check. Python standard library only.

Usage: python check_public_aggregates_v01.py /path/to/PAPER1_COMPONENTS_v02.json
Does not read human records, images, mappings or private inputs; does not write files.
"""
import sys,json,math,hashlib
from pathlib import Path

def check(path):
    raw=Path(path).read_bytes();r=json.loads(raw);n=0
    assert r['population_images']==1497 and r['sample_images']==258 and len(r['variants'])==22
    for name,v in r['variants'].items():
        for pop,p in v['populations'].items():
            assert p['stress_R_scenario']==(53.550000000000004 if pop=='full' else 6.15)
            for view,c in p['constructions'].items():
                th=c['M_Z_T'];S=c['cov_9x9']
                assert len(th)==len(S)==9
                for label,bs in [('overall',[0,1,2]),('H1',[0]),('H2',[1]),('H3',[2])]:
                    d=c['diagnostic'][label];g=[0.]*9;R=0.;defined=True
                    for b in bs:
                        j=3*b;M,Z,T=th[j:j+3]
                        if Z<=0 or T<=0:defined=False;break
                        R+=M*T/Z;g[j:j+3]=[T/Z,-M*T/(Z*Z),M/Z]
                    if not defined:assert d['point'] is None;continue
                    V=sum(g[i]*S[i][j]*g[j] for i in range(9) for j in range(9))
                    assert math.isclose(R,d['point'],rel_tol=1e-10,abs_tol=1e-9)
                    assert math.isclose(V,d['variance'],rel_tol=1e-10,abs_tol=1e-8)
                    assert math.isclose(R+1.645*math.sqrt(max(V,0)),d['Va_z'],rel_tol=1e-10)
                    for e in ('Ea','Eb'):
                        con='envelope' if view.startswith('envelope') else 'primary'
                        BV=sum(p['reference'][con]['B_V'][b] for b in bs);den=sum(p['reference']['primary']['B_V'][b] for b in bs)
                        D=sum(p['duplicates'][con]['DU_'+e][b] for b in bs)
                        for method,x in c['reference_endpoints'][label][e].items():
                            u={'point':d['point'],'Va_z':d['Va_z'],'Va_log':d['Va_log'],'Vb':c['bootstrap_within_stratum'][label]['upper'],'Vb2':c['bootstrap_whole_drive'][label]['upper']}[method]
                            if x is None:assert u is None or den==0;continue
                            assert abs(x['lower']-(BV-D))<1e-8 and abs(x['upper']-(BV+u))<1e-8
                            assert abs(x['relative_half_width']-(D+u)/(2*den))<1e-9;n+=1
                for mode in ('bootstrap_within_stratum','bootstrap_whole_drive'):
                    for x in c[mode].values():assert (x['upper'] is None)==(x['n_undefined']>0)
            for label,bs in [('overall',[0,1,2]),('H1',[0]),('H2',[1]),('H3',[2])]:
                den=sum(p['reference']['primary']['B_V'][b] for b in bs)
                for e in ('Ea','Eb'):
                    for method,outer in p['outer_envelope'][label][e].items():
                        xs=[c['reference_endpoints'][label][e][method] for c in p['constructions'].values()]
                        if outer is None:assert any(x is None for x in xs);continue
                        assert outer['lower']==min(x['lower'] for x in xs) and outer['upper']==max(x['upper'] for x in xs)
                        assert math.isclose(outer['relative_half_width'],(outer['upper']-outer['lower'])/(2*den))
    f=r['variants']['frozen']['populations']['full'];g=r['variants']['frozen']['populations']['gate']
    assert f['constructions']['primary_lower']['finds_by_band']==[38,63,29]
    assert f['constructions']['primary_upper']['finds_by_band']==[38,66,29]
    assert g['constructions']['primary_lower']['finds_by_band']==[10,14,14]
    assert sum(f['reference']['primary']['B_V'])==380 and sum(g['reference']['primary']['B_V'])==63
    return {'status':'PASS','endpoint_checks':n,'variants':22,'sha256':hashlib.sha256(raw).hexdigest(),'scope':'Aggregate arithmetic only; no independent measurements, raw-record rejoin or coverage validation.'}
if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
