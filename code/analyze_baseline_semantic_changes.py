"""Post-hoc event ledger, not a selector or a new model experiment.

Relates official fixed-default/semantic/inner-selected OOF predictions to GT.
Signals support numerical failure-stage descriptions, never semantic causes.
"""
import csv
from collections import Counter
import json
from pathlib import Path
import numpy as np
import run_baseline_semantic_readout as R
from optimized_locator import spans, interval_iou, decode

OUT=R.OUT/'event-change-ledger'


def matched_events(pred,events,threshold=.5):
    used=set()
    for span in pred:
        overlaps=[(interval_iou(span,event),j) for j,event in enumerate(events) if j not in used]
        if overlaps:
            overlap,j=max(overlaps,key=lambda x:(x[0],-x[1]))
            if overlap>=threshold:used.add(j)
    return used


def evidence_stage(probability,video,fps,event,pred,config):
    s,e=event; raw=float(probability[s:e+1].max());gated=raw*float(video)**config['video_strength']
    if raw<config['threshold']:return 'frame_evidence_below_high_seed'
    if gated<config['threshold']:return 'video_attenuation_drops_event_peak_below_seed'
    if not any(interval_iou(event,p)>0 for p in pred):return 'seed_present_but_no_surviving_overlap'
    return 'surviving_overlap_needs_boundary_or_structure'


def mapping(report,name):
    rows=report[name]['predictions']
    return {row['index']:[tuple(x) for x in row['segments']] for row in rows}


def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    if OUT.exists():raise FileExistsError('refuse post-hoc ledger overwrite')
    protocol=R.check_protocol(inputs=True);manifest,records,_,_=R.feature_records()
    report=json.loads((R.RUN/'report.json').read_text('utf-8'))
    predictions={'default_control':mapping(report,'fixed_default_control'),
                 'semantic_content':mapping(report,'fixed_semantic_content_diagnostic_not_selected_deployment'),
                 'nested_primary':mapping(report,'primary_nested_representation_selection')}
    probabilities={name:R.Probabilities({},{}) for name in predictions}
    for fold in range(5):
        caches={};meta=None
        for recipe,_ in R.B.MEMBERS:
            _,outer,meta=R.load_fold(fold,recipe,records);caches[recipe]=outer
        ids=meta['validation'];base=R.blend_probabilities(R.FIXED,caches,ids)
        members,_=R.load_job(f's{fold}_outer',records)
        enabled=R.fuse(members,caches['rgb_motion_tcn'],ids,records)
        selected=next(row for row in report['folds'] if row['fold']==fold)['chosen']['mode']
        for name,scores in (('default_control',base),('semantic_content',enabled),
                            ('nested_primary',enabled if selected=='semantic_content' else base)):
            probabilities[name].frame.update(scores.frame);probabilities[name].video.update(scores.video)
            decoded=R.decode_scores(records,ids,scores,protocol['default_decoder'])
            if any(decoded[i]!=predictions[name][i] for i in ids):raise ValueError('ledger scores differ from official intervals')
    OUT.mkdir();videos=[];events=[];transition=Counter();stages={m:Counter() for m in predictions}
    for i,r in enumerate(records):
        gt=spans(r['labels']);matches={m:matched_events(predictions[m][i],gt) for m in predictions}
        row={'index':i,'name':r['name'],'source_sha256':r['sha256'],'fps':r['fps'],'frames':r['frames'],
             'normal':not bool(gt),'event_count':len(gt),'semantic_cause':'unknown_not_inferred_from_scores'}
        for mode in predictions:
            row[mode+'_segments']=json.dumps(predictions[mode][i])
            row[mode+'_matched_events_iou_0_5']=len(matches[mode])
            row[mode+'_empty']=not bool(predictions[mode][i])
        row['matched_event_delta_primary_vs_default']=len(matches['nested_primary'])-len(matches['default_control'])
        row['matched_event_delta_fixed_semantic_vs_default']=len(matches['semantic_content'])-len(matches['default_control'])
        videos.append(row)
        for j,event in enumerate(gt):
            er={'index':i,'name':r['name'],'event':j,'start_frame':event[0],'end_frame':event[1],
                'duration_seconds':(event[1]-event[0]+1)/r['fps'],'semantic_cause':'unknown_not_inferred_from_scores'}
            for mode in predictions:
                pred=predictions[mode][i];p=probabilities[mode].frame[i];vp=probabilities[mode].video[i]
                er[mode+'_matched_iou_0_5']=j in matches[mode]
                er[mode+'_best_iou']=max([interval_iou(event,span) for span in pred],default=0.)
                er[mode+'_raw_peak']=float(p[event[0]:event[1]+1].max())
                er[mode+'_effective_peak']=er[mode+'_raw_peak']*float(vp)**protocol['default_decoder']['video_strength']
                er[mode+'_diagnostic_stage']=evidence_stage(p,vp,r['fps'],event,pred,protocol['default_decoder'])
                if j not in matches[mode]:stages[mode][er[mode+'_diagnostic_stage']]+=1
            for mode in ('semantic_content','nested_primary'):
                before=er['default_control_matched_iou_0_5'];after=er[mode+'_matched_iou_0_5']
                transition[mode+('/both_match' if before and after else '/lost' if before else '/gained' if after else '/both_miss')]+=1
            events.append(er)
    write_csv(OUT/'videos.csv',videos);write_csv(OUT/'events.csv',events)
    summary={'role':'posthoc_diagnostic_not_selection_or_new_validation','rows':len(records),'events':len(events),
        'event_transition_counts':dict(transition),'unmatched_failure_stage_counts':{k:dict(v) for k,v in stages.items()},
        'semantic_causes':'unknown; confidence stage is not proof of physical cause',
        'matched_uses_prediction_order_greedy_iou_0_5':True,'best_iou_is_individual_diagnostic_not_event_recall':True,
        'official_report_sha256':R.B.digest(R.RUN/'report.json'),'analysis_source_sha256':R.B.digest(Path(__file__))}
    R.B.write_new(OUT/'diagnostic_summary.json',summary)
    # Balanced evidence: show one gain, one loss and one persistent miss. These
    # are disclosed OOF diagnostics, not a curated claim of demo accuracy.
    selected=[]
    for kind,condition,sort_key in (
        ('gain',lambda e:not e['default_control_matched_iou_0_5'] and e['semantic_content_matched_iou_0_5'],lambda e:e['semantic_content_best_iou']-e['default_control_best_iou']),
        ('loss',lambda e:e['default_control_matched_iou_0_5'] and not e['semantic_content_matched_iou_0_5'],lambda e:e['default_control_best_iou']-e['semantic_content_best_iou']),
        ('persistent_miss',lambda e:not e['default_control_matched_iou_0_5'] and not e['semantic_content_matched_iou_0_5'],lambda e:-e['duration_seconds'])):
        possible=[e for e in events if condition(e)]
        if possible:
            chosen=max(possible,key=lambda e:(sort_key(e),-e['index'],-e['event']))
            selected.append({'kind':kind,**chosen})
    R.B.write_new(OUT/'inspected_cases.json',{'role':'disclosed_balanced_diagnostic_cases_not_demo_test',
        'cases':selected,'interpretation':'images show content only; failure mechanism must not be guessed from aggregate metrics'})
    make_images(manifest,records,probabilities,predictions,selected,protocol['default_decoder'])
    print(json.dumps(summary),flush=True)


def make_images(manifest,records,probabilities,predictions,cases,config):
    import cv2
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    width=420;height=280
    sheet=Image.new('RGB',(3*width,len(cases)*height),'#f4f5f7');draw=ImageDraw.Draw(sheet)
    fig,axes=plt.subplots(len(cases),1,figsize=(12,3.2*len(cases)),squeeze=False,layout='constrained')
    for row,case in enumerate(cases):
        i=case['index'];r=records[i];source=Path(manifest['rows'][i]['input_path'])
        if str(source).startswith('C:'):source=Path('/mnt/c')/str(source)[3:].replace('\\','/')
        if R.B.digest(source)!=r['sha256']:raise ValueError('inspection pixel source changed')
        s,e=case['start_frame'],case['end_frame'];frames=[max(0,s-1),(s+e)//2,min(r['frames']-1,e+1)]
        capture=cv2.VideoCapture(str(source))
        try:
            for col,frame in enumerate(frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=capture.read()
                if not ok:raise ValueError('could not inspect requested case frame')
                image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB));image.thumbnail((width-12,height-58))
                sheet.paste(image,(col*width+6,row*height+50))
                draw.text((col*width+6,row*height+7),f"{case['kind']} / row {i} / frame {frame} / GT {s}:{e}",fill='black')
                draw.text((col*width+6,row*height+25),f"old IoU {case['default_control_best_iou']:.3f} / content IoU {case['semantic_content_best_iou']:.3f}",fill='black')
        finally:capture.release()
        ax=axes[row,0];time=np.arange(r['frames'])/r['fps']
        ax.fill_between(time,0,1,where=np.asarray(r['labels'],bool),color='#22c55e',alpha=.18,label='GT')
        for mode,color in (('default_control','#2563eb'),('semantic_content','#f97316')):
            scores=probabilities[mode];effective=scores.frame[i]*scores.video[i]**config['video_strength']
            ax.plot(time,effective,color=color,label=mode)
        ax.axhline(config['threshold'],color='#666',ls='--',lw=1,label='high seed')
        ax.axhline(config['threshold']*config['low_ratio'],color='#999',ls=':',lw=1,label='low connect')
        ax.set(ylim=(0,1),xlabel='seconds',ylabel='effective probability',title=f"{case['kind']} | row {i}: {r['name']} | GT {s}:{e}")
        ax.legend(ncol=5,fontsize=8,loc='upper right')
    with (OUT/'balanced_case_contact_sheet.png').open('xb') as f:sheet.save(f,format='PNG')
    with (OUT/'balanced_case_timelines.png').open('xb') as f:fig.savefig(f,format='png',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
