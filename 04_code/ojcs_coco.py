"""Frozen COCO known-deletion construction; no detector fitting or selection."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, urllib.request, zipfile
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NS='PROGRAMMEB_OJCS_20261110_COCO_FRAME_V1'
SCENARIOS=('uniform','height_dependent','rare')
ANNOTATION_URL='https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMAGE_URL='https://images.cocodataset.org/zips/val2017.zip'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    t=p.with_suffix(p.suffix+'.partial')
    with t.open('xb') as f:f.write(canonical(x));f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def fetch(url,p):
    p=Path(p)
    if p.exists():return
    p.parent.mkdir(parents=True,exist_ok=True);part=p.with_suffix(p.suffix+'.download')
    if part.exists():raise RuntimeError('Incomplete download preserved: '+str(part))
    with urllib.request.urlopen(url,timeout=45) as response,part.open('xb') as out:
        for block in iter(lambda:response.read(1024*1024),b''):out.write(block)
        out.flush();os.fsync(out.fileno())
    os.replace(part,p)

def rank(tag,value):return hashlib.sha256(f'{NS}|{tag}|{value}'.encode()).digest()
def select_images(images,n=1000):
    ids=[int(x['id']) for x in images]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image ID')
    # Protocol image selection is SHA256(namespace|image_id), with no annotation input.
    return sorted(ids,key=lambda i:(hashlib.sha256(f'{NS}|{i}'.encode()).digest(),i))[:n]

def clipped_box(a,image):
    x,y,w,h=map(float,a['bbox'])
    if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0:return None
    x0=max(0.,x);y0=max(0.,y);x1=min(float(image['width']),x+w);y1=min(float(image['height']),y+h)
    if x1<=x0 or y1<=y0:return None
    return [x0,y0,x1,y1]

def probability(scenario,height):
    if scenario=='uniform':return .20
    if scenario=='rare':return .005
    if scenario=='height_dependent':return .50 if height<40 else (.20 if height<80 else .05)
    raise ValueError(scenario)

def deleted(scenario,a):
    # Compare integer hash to floor(p*2**256), avoiding platform RNG dependence.
    from fractions import Fraction
    p=Fraction(str(probability(scenario,a['box'][3]-a['box'][1])))
    return int.from_bytes(rank('deletion|'+scenario,a['id']),'big') < (p.numerator*(1<<256))//p.denominator

def prepare(archive,out):
    out=Path(out);cohort_path=out/'COHORT.json';target_path=out/'TARGETS.json'
    if Path(archive).suffix=='.json':
        with Path(archive).open(encoding='utf-8') as f:data=json.load(f)
    else:
        with zipfile.ZipFile(archive) as z:
            with z.open('annotations/instances_val2017.json') as f:data=json.load(f)
    images={int(i['id']):i for i in data['images']}
    if len(images)!=5000:raise ValueError('Expected official 5000-image validation split')
    chosen=select_images(data['images'])
    cohort={'namespace':NS,'created_utc':stamp(),'annotation_input_name':Path(archive).name,'annotation_archive_sha256':sha(archive),
            'selection':'SHA256(namespace|image_id), then image_id; before processing object annotations',
            'image_ids':chosen,'images':[{k:images[i][k] for k in ('id','width','height','file_name')} for i in chosen]}
    if cohort_path.exists():
        old=json.loads(cohort_path.read_text())
        if old['image_ids']!=chosen or old['annotation_archive_sha256']!=cohort['annotation_archive_sha256']:raise ValueError('Cohort drift')
        cohort=old
    else:write_new(cohort_path,cohort)
    # The cohort receipt is committed BEFORE processing any annotation outcomes.
    if target_path.exists():raise FileExistsError('Targets already frozen; reuse them')
    people=[c['id'] for c in data['categories'] if c['name']=='person']
    if people!=[1]:raise ValueError('Unexpected person category')
    byimage={i:[] for i in chosen};seen=set()
    for a in data['annotations']:
        if a['image_id'] not in byimage or a['category_id']!=1:continue
        if a['id'] in seen:raise ValueError('Duplicate annotation ID')
        seen.add(a['id']);box=clipped_box(a,images[a['image_id']])
        if box is None:continue
        byimage[a['image_id']].append({'id':int(a['id']),'box':box,'crowd':bool(a.get('iscrowd',0))})
    records=[]
    for i in chosen:
        anns=sorted(byimage[i],key=lambda a:a['id'])
        eligible=[a for a in anns if not a['crowd'] and a['box'][3]-a['box'][1]>=25]
        selected=sorted(eligible,key=lambda a:(rank('target',a['id']),a['id']))[:20]
        deletion_ids={s:sorted(a['id'] for a in selected if deleted(s,a)) for s in SCENARIOS}
        records.append({'image_id':i,'annotations':anns,'target_ids':sorted(a['id'] for a in selected),'deleted_ids':deletion_ids})
    result={'schema':'OJCS_COCO_TARGETS_V1','cohort_sha256':sha(cohort_path),'created_utc':stamp(),
            'target_cap':20,'scenarios':list(SCENARIOS),'records':records,
            'reference_rule':'All retained valid person boxes remain matching references, including non-target and crowd boxes; only non-crowd height>=25 hash-capped targets can be deleted.'}
    write_new(target_path,result)
    print(json.dumps({'status':'COHORT_AND_TARGETS_FROZEN','images':len(chosen),'cohort_sha256':sha(cohort_path),'targets_sha256':sha(target_path),'no_performance_analysis':True}),flush=True)

def matching(detections,references,threshold=.5):
    from scipy.optimize import linear_sum_assignment
    a=np.asarray(detections,dtype=float).reshape(-1,4);b=np.asarray(references,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return []
    left=np.maximum(a[:,None,:2],b[None,:,:2]);right=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(right-left,0),axis=2)
    aa=np.prod(a[:,2:]-a[:,:2],axis=1);bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    iou=inter/(aa[:,None]+bb[None,:]-inter);valid=iou>=threshold
    # A single additional valid edge dominates the total possible IoU difference.
    weight=np.where(valid,min(len(a),len(b))+1+iou,0)
    rows,cols=linear_sum_assignment(weight,maximize=True)
    return [(int(i),int(j),float(iou[i,j])) for i,j in zip(rows,cols) if valid[i,j]]

def build_frames(directory):
    from ojcs_engine import ROOT as _root
    from paper2_generators_v01 import PublicFrame
    d=Path(directory);cohort=json.loads((d/'COHORT.json').read_text());targets=json.loads((d/'TARGETS.json').read_text())
    if targets['cohort_sha256']!=sha(d/'COHORT.json'):raise ValueError('Target cohort mismatch')
    if [x['image_id'] for x in targets['records']]!=cohort['image_ids']:raise ValueError('Target order mismatch')
    detector=json.loads((d/'DETECTOR_COMPLETE.json').read_text())
    frames={};ledger=[]
    for s in SCENARIOS:
        values=[]
        for item in targets['records']:
            i=item['image_id'];p=d/'detections'/f'{i:012}.json';record=json.loads(p.read_text())
            if sha(p)!=detector['outputs'][p.name] or record['image_id']!=i:raise ValueError('Detector checkpoint mismatch')
            boxes=record['boxes'];gone=set(item['deleted_ids'][s]);refs=[a for a in item['annotations'] if a['id'] not in gone]
            pairs=matching(boxes,[a['box'] for a in refs]);matched={a for a,b,c in pairs}
            un=[b for j,b in enumerate(boxes) if j not in matched];heights=[b[3]-b[1] for b in un]
            bins=[sum(25<=h<40 for h in heights),sum(40<=h<80 for h in heights),sum(h>=80 for h in heights)]
            original=sum(not a['crowd'] and a['box'][3]-a['box'][1]>=25 for a in refs)
            values.append([len(un),original,len(boxes),*bins,len(pairs),len(gone)])
            ledger.append({'scenario':s,'image_id':i,'deleted':len(gone),'detector':len(boxes),'eligible_retained':original,'all_retained_person_boxes':len(refs),'matched':len(pairs),'unmatched':len(un),'matching':pairs})
        a=np.asarray(values,dtype=int);n=len(a)
        pub=PublicFrame(a[:,:6].astype(float),a[:,1],a[:,2],a[:,0],a[:,6],np.zeros(n,dtype=int),np.full(n,20),np.array([]),np.array([]),np.array([]))
        path=d/f'FRAME_{s}.npz'
        with path.open('xb') as f:np.savez_compressed(f,**pub.__dict__,omitted=a[:,7])
        frames[s]={'path':path.name,'sha256':sha(path),'frame_id':NS+'|'+s}
    write_new(d/'MATCHING_LEDGER.json',ledger)
    write_new(d/'FRAMES_COMPLETE.json',{'cohort_sha256':sha(d/'COHORT.json'),'targets_sha256':sha(d/'TARGETS.json'),'detector_sha256':sha(d/'DETECTOR_COMPLETE.json'),'frames':frames,'matching_ledger_sha256':sha(d/'MATCHING_LEDGER.json')})
    print(json.dumps({'status':'FRAMES_FROZEN','images':len(cohort['image_ids']),'scenarios':len(frames)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);p.add_argument('--directory',required=True);p.add_argument('--archive');a=p.parse_args()
    if a.action=='prepare':prepare(a.archive,a.directory)
    else:build_frames(a.directory)
