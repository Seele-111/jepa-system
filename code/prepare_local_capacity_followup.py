"""Generate actionable review / paired-data templates, never alter labels."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/baseline-targeted-local-interaction-repaired-20261003"

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def csv_new(path,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ev=read(OUT/"events.json");g=read(OUT/"geometry_inventory.json");vid=read(OUT/"video_support.json")
    rows=read(ROOT/"output/algorithm-opt-2026-10-02-v2/dataset_manifest.json")["rows"]
    queue=[]
    for e in ev:
        if e["category"]=="matched_iou_0_5":continue
        i=e["index"]
        # Mechanisms remain unresolved; do not manufacture ROI or cause labels.
        priority=1 if not e["pure_anchor_indices"] else 2 if e["category"]=="frame_evidence_below_high_seed" else 3
        queue.append({"priority":priority,"row_index":i,"event":e["event"],"source_name":e["name"],"source_sha256":e["sha256"],
                      "start_frame":e["start_frame"],"end_frame":e["end_frame"],"failure_stage":e["category"],
                      "pure_V_anchor_count":len(e["pure_anchor_indices"]),"V_source_area_fraction":g[i]["v_source_area_fraction"],
                      "median_V_anchor_spacing_seconds":e["median_anchor_spacing_seconds"],
                      "review_required":"locate actual erroneous object and compare retained crop; check short boundary evidence",
                      "physical_cause_reviewed":"","roi_frame_id":"","roi_x0_source_normalized":"","roi_y0_source_normalized":"",
                      "roi_x1_source_normalized":"","roi_y1_source_normalized":"","erroneous_object_inside_V_crop":"",
                      "reference_normal_scene_or_pair":"","annotator":"","second_review_agrees":"","notes":""})
    queue.sort(key=lambda r:(r["priority"],r["row_index"],r["event"]))
    csv_new(OUT/"targeted_event_review_queue.csv",queue)
    normal=[]
    for v in vid:
        if not v["is_normal"]:continue
        i=v["index"]
        normal.append({"row_index":i,"source_name":v["name"],"source_sha256":v["sha256"],
                       "review_required":"confirm normal label, document high motion/occlusion/camera cases; do not relabel to improve score",
                       "semantic_category_reviewed":"","similar_anomaly_source_name":"","annotator":"","notes":""})
    csv_new(OUT/"targeted_normal_review_queue.csv",normal)
    categories=[("ball_contact",4),("fast_hand_tool",3),("body_sports_occlusion",3),("rolling_bouncing_object",3),("camera_motion_lighting",2)]
    plan=[];pair=0
    for cat,count in categories:
        for _ in range(count):
            pair+=1
            for expected in ["normal","anomaly"]:
                plan.append({"slot_id":f"pair{pair:02d}_{expected}","pair_group":f"pair{pair:02d}","planned_category":cat,
                             "expected_label_NOT_actual_annotation":expected,"reserve_as_frozen_independent_test":True,
                             "needs_short_event_le_0_5sec":expected=="anomaly" and pair<=8,
                             "needs_multi_event":expected=="anomaly" and pair in [1,2,3,9,10,11],
                             "different_generator_and_scene_required":True,"status":"not_collected",
                             "source_absolute_path":"","source_sha256":"","source_family":"","near_duplicate_group":"",
                             "annotations_absolute_path":"","annotator_1":"","annotator_2":"","adjudication_complete":"",
                             "frozen_before_model_selection":"","notes":""})
    csv_new(OUT/"paired_frozen_test_collection_template.csv",plan)
    write={"review_events":len(queue),"review_normal_videos":len(normal),"planned_new_test_slots":len(plan),
           "planned_pairs":15,"short_anomaly_slots":8,"multi_event_anomaly_slots":6,
           "none_collected_or_validated_yet":True,"no_new_labels_created":True,
           "policy":"Pair/source/near-duplicate groups cannot cross train/test; frozen final test labels never used to tune capacity or decoder."}
    (OUT/"followup_worklist.json").write_text(json.dumps(write,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(write))

if __name__=="__main__":main()
