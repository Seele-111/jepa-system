"""Fast per-video sufficient statistics; preserves inclusive greedy legacy metrics."""
from __future__ import annotations
import numpy as np
from optimized_locator import spans


class VideoTruth:
    def __init__(self, labels):
        self.labels=np.asarray(labels,dtype=bool).reshape(-1)
        self.events=spans(self.labels)
        self.prefix=np.r_[0,np.cumsum(self.labels,dtype=np.int64)]
        self.normal=not self.events


def video_statistics(predictions, truth):
    pred=[(int(s),int(e)) for s,e in predictions];n=len(truth.labels)
    event=[]
    for threshold in (.3,.5):
        used=set();tp=0
        for s,e in pred:
            best=-1;best_iou=0.
            for j,(gs,ge) in enumerate(truth.events):
                if j in used:continue
                intersection=max(0,min(e,ge)-max(s,gs)+1)
                union=max(1,e-s+1+ge-gs+1-intersection)
                overlap=intersection/union
                if overlap>best_iou:best_iou=overlap;best=j
            if best>=0 and best_iou>=threshold:used.add(best);tp+=1
        event.extend([tp,len(pred)-tp,len(truth.events)-tp])
    intervals=sorted((max(0,s),min(n-1,e)) for s,e in pred if max(0,s)<=min(n-1,e))
    merged=[]
    for s,e in intervals:
        if merged and s<=merged[-1][1]+1:merged[-1]=(merged[-1][0],max(e,merged[-1][1]))
        else:merged.append((s,e))
    frame_tp=sum(int(truth.prefix[e+1]-truth.prefix[s]) for s,e in merged)
    covered=sum(e-s+1 for s,e in merged)
    return np.asarray(event+[frame_tp,covered-frame_tp,int(truth.prefix[-1])-frame_tp,
                        int(truth.normal and bool(pred)),int(not truth.normal and not pred),len(pred)],dtype=np.int64)


def f1(tp,fp,fn):
    return 2*tp/max(1,2*tp+fp+fn)


def utility(stats, normal_count, positive_count):
    s=np.asarray(stats)
    return (.35*f1(*s[:3])+.65*f1(*s[3:6])-.20*s[9]/max(1,normal_count)-.15*s[10]/max(1,positive_count))


def stratified_bootstrap_counts(records, indices, seed, replicates=32):
    groups={0:[],1:[],2:[]}
    for j,i in enumerate(indices):groups[min(2,int(records[i]['event_count']))].append(j)
    rng=np.random.default_rng(seed);counts=np.zeros((replicates,len(indices)),np.int64)
    for group in groups.values():
        if group:
            for r in range(replicates):np.add.at(counts[r],rng.choice(group,len(group),replace=True),1)
    return counts
