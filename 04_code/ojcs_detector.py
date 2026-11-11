"""One frozen pretrained detector pass; checkpoint every image; no training."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'04_code'))
from ojcs_coco import sha,write_new,stamp

def main(max_seconds):
    import torch,torchvision
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    if torch.__version__.split('+')[0]!='2.8.0' or torchvision.__version__.split('+')[0]!='0.23.0':raise RuntimeError('Expected torch 2.8.0 / torchvision 0.23.0')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this fixed pass')
    torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=ROOT/'04_code'/'data'/'coco';cohort=json.loads((d/'COHORT.json').read_text());assets=json.loads((d/'ASSETS_COMPLETE.json').read_text())
    if assets['cohort_sha256']!=sha(d/'COHORT.json'):raise ValueError('Asset cohort mismatch')
    if (d/'DETECTOR_COMPLETE.json').exists():raise RuntimeError('Completed inference must not restart')
    weight=d/assets['weight']['filename']
    if sha(weight)!=assets['weight']['sha256']:raise ValueError('Weight checksum')
    context={'cohort_sha256':sha(d/'COHORT.json'),'assets_sha256':sha(d/'ASSETS_COMPLETE.json'),'detector_source_sha256':sha(__file__),'weight_sha256':sha(weight),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'dtype':'float32','batch_size':1,'score_threshold':.5,'minimum_height':25,'person_label':1,'min_size':800,'max_size':1333,'box_nms_thresh':.5,'box_detections_per_img':100,'tf32':False,'cudnn_deterministic':True}
    context_hash=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    context_path=d/'DETECTOR_CONTEXT.json'
    if context_path.exists():
        if json.loads(context_path.read_text())!=context:raise ValueError('Detector resume context changed')
    else:write_new(context_path,context)
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=91)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval().cuda()
    (d/'detections').mkdir(exist_ok=True);images={r['image_id']:r for r in assets['images']};outputs={};began=time.monotonic();inference=0.
    with torch.inference_mode():
        for n,item in enumerate(cohort['images'],1):
            if time.monotonic()-began>max_seconds:raise TimeoutError('Fixed detector deadline')
            i=item['id'];p=d/'images'/item['file_name'];out=d/'detections'/f'{i:012}.json'
            if sha(p)!=images[i]['sha256']:raise ValueError('Image checksum')
            if out.exists():
                record=json.loads(out.read_text())
                if record['context_sha256']!=context_hash or record['image_sha256']!=images[i]['sha256']:raise ValueError('Detection resume mismatch')
            else:
                with Image.open(p) as im:
                    if im.size!=(item['width'],item['height']):raise ValueError('Image size')
                    tensor=pil_to_tensor(im.convert('RGB')).float().div_(255).cuda()
                torch.cuda.synchronize();start=time.perf_counter();prediction=model([tensor])[0];torch.cuda.synchronize();elapsed=time.perf_counter()-start;inference+=elapsed
                keep=(prediction['labels']==1)&(prediction['scores']>=.5)&((prediction['boxes'][:,3]-prediction['boxes'][:,1])>=25)
                boxes=prediction['boxes'][keep].cpu().tolist();scores=prediction['scores'][keep].cpu().tolist()
                record={'image_id':i,'image_sha256':images[i]['sha256'],'context_sha256':context_hash,'boxes':boxes,'scores':scores,'inference_seconds':elapsed}
                write_new(out,record)
            outputs[out.name]=sha(out)
            if n%25==0:
                progress={'utc':stamp(),'completed':n,'total':1000,'elapsed_seconds':time.monotonic()-began,'state':'RUNNING'}
                temp=d/'DETECTOR_PROGRESS.tmp';temp.write_text(json.dumps(progress));os.replace(temp,d/'DETECTOR_PROGRESS.json');print(json.dumps(progress),flush=True)
    write_new(d/'DETECTOR_COMPLETE.json',{'utc':stamp(),'context_sha256':context_hash,'context_file_sha256':sha(context_path),'outputs':outputs,'images':len(outputs),'inference_seconds_this_launch':inference,'wall_seconds_this_launch':time.monotonic()-began})
    print(json.dumps({'status':'DETECTOR_COMPLETE','images':len(outputs)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-seconds',type=int,default=5400);a=p.parse_args()
    if not 1<=a.max_seconds<=5400:raise SystemExit('Detector execution cap is 90 minutes, within 2-hour pod ceiling')
    main(a.max_seconds)
