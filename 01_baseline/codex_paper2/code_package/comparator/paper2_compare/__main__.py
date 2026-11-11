"""Portable CLI. The full-run command always requires separately bound consent."""
import argparse,json
from pathlib import Path
from .storage import validate_manifest,load,new_json,owned,file_sha

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('validate');a.add_argument('--manifest',required=True)
    a=sub.add_parser('fixtures');a.add_argument('--output',required=True);a.add_argument('--work-root',required=True)
    a=sub.add_parser('profile');a.add_argument('--output',required=True);a.add_argument('--work-root',required=True)
    a=sub.add_parser('run')
    for k in ('manifest','approval','output','work-root'):a.add_argument('--'+k,required=True)
    for name in ('verify','summarize'):
        a=sub.add_parser(name);a.add_argument('--manifest',required=True);a.add_argument('--input',required=True)
        if name=='summarize':a.add_argument('--output',required=True);a.add_argument('--work-root',required=True)
    args=p.parse_args()
    if args.command=='validate':
        manifest,config=validate_manifest(args.manifest)
        result={'status':'VALIDATED_RELEASE_ONLY','files':len(manifest['files']),'execution_authorized':False}
    elif args.command=='run':
        from .engine import run
        run(args.manifest,args.approval,args.output,args.work_root);return
    elif args.command in ('fixtures','profile'):
        from .fixtures import fixture_checks,profile
        output=owned(args.output,args.work_root)
        result=(fixture_checks if args.command=='fixtures' else profile)(output,args.work_root)
    else:
        from .review import summarize
        manifest,config=validate_manifest(args.manifest)
        result=summarize(args.input,file_sha(args.manifest),config,strict=True,replay=True)
        if args.command=='summarize':new_json(owned(args.output,args.work_root),result)
        else:
            expected=load(Path(args.input)/'FINAL_SUMMARY.json')
            if result!=expected:raise ValueError('saved summary differs from raw records')
            result={'status':'RAW_SUMMARIES_REPRODUCED','counts':result['counts']}
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
