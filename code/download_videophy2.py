#!/usr/bin/env python3
"""Download videophy2_test videos to /mnt/e/jepa-test/"""
import csv, os, subprocess, sys, time

CSV_PATH = "/home/zzy/.cache/huggingface/hub/datasets--videophysics--videophy2_test/snapshots/90b81ffa54f565d9e40e83b7a2c247ba1dccfa2b/videophy2_test.csv"
OUT_DIR = "/mnt/e/jepa-test"

os.makedirs(OUT_DIR, exist_ok=True)

with open(CSV_PATH) as f:
    urls = [row['video_url'] for row in csv.DictReader(f)]

print(f"Total: {len(urls)} videos")
existing = set(os.listdir(OUT_DIR))
todo = []
for url in urls:
    fname = url.split('/')[-1].split('?')[0]
    if fname not in existing:
        todo.append((url, fname))

print(f"New: {len(todo)}, Already have: {len(urls)-len(todo)}")

success = 0
fail = 0
for i, (url, fname) in enumerate(todo):
    out = os.path.join(OUT_DIR, fname)
    print(f"[{i+1}/{len(todo)}] {fname[:60]}...", end=' ', flush=True)
    try:
        r = subprocess.run([
            'timeout', '45', 'wget', '-q', '--timeout=15', '--tries=1',
            '-O', out, url
        ], timeout=60, capture_output=True)
        if r.returncode == 0 and os.path.getsize(out) > 0:
            print("OK")
            success += 1
        else:
            print(f"FAIL ({r.returncode})")
            fail += 1
            os.remove(out) if os.path.exists(out) else None
    except Exception as e:
        print(f"ERR: {e}")
        fail += 1

    if (i+1) % 50 == 0:
        print(f"  --- Progress: {success} ok, {fail} fail ---")

print(f"\nDone: {success} downloaded, {fail} failed, {len(urls)-len(todo)} cached")
