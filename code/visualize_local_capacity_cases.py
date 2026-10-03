"""Disclosed numeric case selection and source-coordinate visual evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"output/algorithm-opt-2026-10-02/corrected_jepa_features_v2"
SOURCES=Path("C:/Users/admin/Desktop/测试")

def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output
    require=lambda ok,msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require(not (out/"case_contact_sheet.png").exists(),"no overwrite")
    protocol=read(out/"protocol.json");events=read(out/"events.json");videos=read(out/"video_support.json");geo=read(out/"geometry_inventory.json")
    names=protocol["features"];j=names.index("v_strength_change")
    rows=read(ROOT/"output/algorithm-opt-2026-10-02-v2/dataset_manifest.json")["rows"]
    raw={e.get("name",e.get("video_name")):e for e in read(RAW/"manifest.json")["videos"]}
    official=read(ROOT/"output/baseline-targeted-semantic-readout/controlled-readout/report.json")
    pred={e["index"]:e["segments"] for e in official["fixed_default_control"]["predictions"]}
    arrays=np.load(out/"anchor_features.npz",allow_pickle=False)
    def score(e):
        i=e["index"];ids=e["pure_anchor_indices"];a=arrays[f"values_{i}"][ids,j];m=arrays[f"valid_{i}"][ids,j].astype(bool)
        return float(a[m].mean()) if m.any() else None
    low=[e for e in events if e["category"]=="frame_evidence_below_high_seed" and score(e) is not None]
    gated=[e for e in events if e["category"]=="video_attenuation_drops_event_peak_below_seed" and score(e) is not None]
    missing=[e for e in events if e["category"]=="surviving_overlap_needs_boundary_or_structure" and not e["pure_anchor_indices"]]
    selected=[]
    for kind,ev in [("LOW strongest relation",max(low,key=lambda e:(score(e),-e["index"]))),
                    ("LOW weakest relation",min(low,key=lambda e:(score(e),e["index"]))),
                    ("GATED median relation",sorted(gated,key=lambda e:(score(e),e["index"]))[len(gated)//2]),
                    ("BOUNDARY no pure tubelet",missing[0])]:
        selected.append({"kind":kind,**ev,"selection_score":score(ev)})
    normal=[]
    for v in videos:
        if v["is_normal"]:
            i=v["index"];a=arrays[f"values_{i}"][:,j];m=arrays[f"valid_{i}"][:,j].astype(bool)
            normal.append((float(a[m].max()),i))
    normal.sort()
    for kind,(s,i) in [("NORMAL strongest relation",normal[-1]),("NORMAL median relation",normal[len(normal)//2])]:
        selected.append({"kind":kind,"index":i,"name":rows[i]["name"],"category":"normal","selection_score":s})
    cw,ch=440,310
    sheet=Image.new("RGB",(cw*3,ch*len(selected)),"#fafafa");draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",14)
    fig,axes=plt.subplots(len(selected),1,figsize=(13,3*len(selected)),layout="constrained")
    receipt=[]
    for rr,case in enumerate(selected):
        i=case["index"];r=rows[i];g=geo[i];path=SOURCES/r["name"]
        require(sha(path)==r["sha256"],"source changed")
        tubelets=arrays[f"tubelets_{i}"];centers=arrays[f"centers_{i}"]
        if case["category"]=="normal":
            valid=arrays[f"valid_{i}"][:,j].astype(bool);vals=np.where(valid,arrays[f"values_{i}"][:,j],-np.inf)
            middle=int(np.argmax(vals));ks=sorted(set([max(0,middle-1),middle,min(len(centers)-1,middle+1)]))
        else:
            s,e=case["start_frame"],case["end_frame"]
            ks=sorted(set(int(np.argmin(abs(centers-f))) for f in [s,(s+e)/2,e]))
        while len(ks)<3: ks.append(ks[-1])
        with np.load(RAW/raw[r["name"]]["raw_directory"]/"signals.npz",allow_pickle=False) as a:
            heat=a["vjepa_raw_heatmaps"];mask=a["vjepa_patch_valid_mask"].astype(bool)&(a["vjepa_patch_counts"]>0)
        cap=cv2.VideoCapture(str(path));actual=[]
        try:
            for cc,k in enumerate(ks[:3]):
                frame=int(tubelets[k,0]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=cap.read()
                require(ok,"inspection decode failed")
                actual.append(frame)
                image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)).convert("RGBA")
                w,h=image.size;overlay=Image.new("RGBA",image.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
                x0,y0,x1,y1=g["v"]["box_xyxy"];vals=heat[k][mask[k]];med=float(np.median(vals));high=float(np.quantile(vals,.95))
                for pr in range(24):
                    for pc in range(24):
                        box=[(x0+(x1-x0)*pc/24)*w,(y0+(y1-y0)*pr/24)*h,(x0+(x1-x0)*(pc+1)/24)*w,(y0+(y1-y0)*(pr+1)/24)*h]
                        if mask[k,pr,pc]:
                            intensity=float(np.clip((heat[k,pr,pc]-med)/max(high-med,1e-8),0,1))
                            if intensity>0: od.rectangle(box,fill=(239,68,68,int(140*intensity)))
                        else: od.rectangle(box,fill=(45,45,45,90))
                image=Image.alpha_composite(image,overlay).convert("RGB")
                imd=ImageDraw.Draw(image)
                imd.rectangle([x0*w,y0*h,x1*w,y1*h],outline="#00d9ff",width=max(2,w//250))
                b=g["i"]["box_xyxy"];imd.rectangle([b[0]*w,b[1]*h,b[2]*w,b[3]*h],outline="#ffbb00",width=max(1,w//350))
                image.thumbnail((cw-12,ch-85));sheet.paste(image,(cc*cw+6,rr*ch+78))
                draw.text((cc*cw+7,rr*ch+7),f"{case['kind']} | row {i}",fill="black",font=font)
                draw.text((cc*cw+7,rr*ch+27),f"frame {frame} | tubelet {tubelets[k].tolist()}",fill="black",font=font)
                gt=f"GT {case['start_frame']}:{case['end_frame']}" if case["category"]!="normal" else "GT normal"
                draw.text((cc*cw+7,rr*ch+47),f"{gt} | observed patches {int(mask[k].sum())}/576",fill="black",font=font)
        finally:cap.release()
        ax=axes[rr];scores=arrays[f"values_{i}"][:,j].copy();valid=arrays[f"valid_{i}"][:,j].astype(bool);scores[~valid]=np.nan
        ax.plot(centers/r["fps"],scores,"o-",label="native V strength x local change",color="#dc2626")
        if case["category"]!="normal": ax.axvspan(case["start_frame"]/r["fps"],(case["end_frame"]+1)/r["fps"],color="#16a34a",alpha=.18,label="annotated target event")
        for z,(s,e) in enumerate(pred[i]): ax.axvspan(s/r["fps"],(e+1)/r["fps"],color="#2563eb",alpha=.18,label="default OOF segment" if z==0 else None)
        ax.set(xlabel="seconds",ylabel="relative local error feature\nNOT anomaly probability",title=f"{case['kind']} | row {i} | displayed frames {actual}")
        ax.legend(loc="upper right",fontsize=8)
        receipt.append({**case,"displayed_frames":actual,"default_OOF_segments":pred[i],"source_sha256_verified":True})
    sheet.save(out/"case_contact_sheet.png");fig.savefig(out/"case_timelines.png",dpi=140);plt.close(fig);arrays.close()
    (out/"inspected_cases.json").write_text(json.dumps({"role":"disclosed_numeric_cases_not_blind_test_not_cherry_picked_demo",
      "selection":"LOW high/low, GATED median, BOUNDARY no-pure-tubelet, NORMAL high/median; selected after diagnostic solely for review",
      "legend":"cyan V crop; yellow I crop; red relative raw patch error; gray unobserved; outside crop NOT observed; frames are actual V tubelet first member",
      "cases":receipt},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","cases":[(c['kind'],c['index']) for c in receipt]}))

if __name__=="__main__":main()
