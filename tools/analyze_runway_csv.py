"""Analyze mapped DLC CSVs locally, retaining source-frame alignment and reports."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from Functions import KinematicsFunctions as k


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path,nargs='+')
    parser.add_argument('--fps',type=float,required=True)
    parser.add_argument('--pixels-per-cm',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cutoff',type=float,default=6.)
    parser.add_argument('--confidence',type=float,default=.6)
    parser.add_argument('--max-gap-ms',type=float,default=20.)
    parser.add_argument('--min-body-speed',type=float,default=.25,
        help='Minimum speed in median toe/body-reference lengths per second; the adaptive tracking-noise floor also applies.')
    parser.add_argument('--min-contact-ms',type=float,default=30.)
    parser.add_argument('--max-step-height',type=float,default=2.)
    parser.add_argument('--max-stride-length',type=float,default=8.)
    parser.add_argument('--drag-clearance',type=float,default=.1)
    parser.add_argument('--direction',choices=['auto','left_to_right','right_to_left'],default='auto')
    parser.add_argument('--joint-interpolation',choices=['none','linear','pchip'],default='none')
    parser.add_argument('--joint-max-gap-ms',type=float,default=20.)
    parser.add_argument('--joint-max-interpolated-percent',type=float,default=10.)
    parser.add_argument('--joint-max-segment-change-percent',type=float,default=35.)
    parser.add_argument('--joint-cutoff',type=float,help='Separate joint/distance cutoff in Hz; otherwise inherit --cutoff.')
    args=parser.parse_args()
    if len({path.stem for path in args.csv})!=len(args.csv):
        parser.error('Input filenames must be unique within an output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    direction={'auto':'auto','left_to_right':False,'right_to_left':True}[args.direction]
    for path in args.csv:
        data,_,_=k.fix_column_names(pd.read_csv(path,header=[1,2]))
        params,coords,phase,_,masks,starts=k.extract_spontaneous_parameters(args.fps,data,args.cutoff,
            pixels_per_cm=args.pixels_per_cm,confidence_threshold=args.confidence,max_gap_s=args.max_gap_ms/1000,
            min_body_speed=args.min_body_speed,min_contact_s=args.min_contact_ms/1000,
            step_height_max_cm=args.max_step_height,stride_length_max_cm=args.max_stride_length,
            drag_clearance_cm=args.drag_clearance,right_to_left=direction,
            joint_interpolation=args.joint_interpolation,joint_max_gap_s=args.joint_max_gap_ms/1000,
            joint_max_interpolated_fraction=args.joint_max_interpolated_percent/100,
            joint_max_segment_change=args.joint_max_segment_change_percent/100,joint_cutoff_f=args.joint_cutoff)
        report=params.attrs['analysis_report']
        report['source_csv']=path.name
        prefix=args.output/path.stem
        params.to_csv(str(prefix)+'_parameters.csv',index=False)
        coords.to_csv(str(prefix)+'_coordinates.csv',index=False)
        Path(str(prefix)+'_analysis_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        arrays=dict(phase=phase,starts=np.asarray(starts,dtype=int))
        arrays.update({f'drag_{i}':mask for i,mask in enumerate(masks)})
        np.savez(str(prefix)+'_events.npz',**arrays)
        print(f'{path.name}: {report["retained_intervals"]}/{report["candidate_intervals"]} complete intervals retained; {len(report["bouts"])} moving bouts',flush=True)


if __name__=='__main__':
    main()
