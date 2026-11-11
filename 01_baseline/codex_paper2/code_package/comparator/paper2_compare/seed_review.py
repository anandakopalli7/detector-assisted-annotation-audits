"""Check integer seed collisions without generating populations or outcomes."""
import hashlib
from .engine import seed

def audit_seeds(config,old):
    seen=set();digest=hashlib.sha256();counts={'new_population':0,'new_audit':0,'old_population_checked':0,'old_audit_checked':0}
    def add(text,kind):
        number=seed(text)
        if number in seen:raise ValueError('new seed collision: '+text)
        seen.add(number);digest.update((text+'|'+str(number)+'\n').encode());counts[kind]+=1
    for m in config['mechanisms']:
        for j in range(config['populations_per_mechanism']):add(f"{config['population_seed_prefix']}|{m}|{j}",'new_population')
        for b in config['budgets']:
            for r in range(config['replicates_per_cell']):
                for stream in config['streams']:add(f"{config['audit_seed_prefix']}|{m}|{b}|{r}|{stream}",'new_audit')
    def oldcheck(text,kind):
        if seed(text) in seen:raise ValueError('overlap with old final seed: '+text)
        counts[kind]+=1
    for m in old['mechanisms']:
        for j in range(old['populations_per_mechanism']):oldcheck(f"{old['population_seed_prefix']}|{m}|{j}",'old_population_checked')
        for b in old['budgets']:
            count=max(old['risk_looks']) if m in old['efficiency_mechanisms'] and b==old['primary_budget'] else old['coverage_repetitions']
            for r in range(count):oldcheck(f"{old['audit_seed_prefix']}|{m}|{b}|{r}",'old_audit_checked')
    return {'status':'PASS','counts':counts,'new_seed_catalog_sha256':digest.hexdigest(),'collisions':0,
        'scope':'Entire old approved maximum seed schedule, even if a risk look was not needed; no RNG draw or population construction',
        'folds':'Direct SHA256 rankings, not PCG64 seeds; ties deterministically broken by unit ID'}
