"""Read the frozen producer's integer-key metadata without rewriting evidence.

JSON restores dictionary keys as strings; the producer hashed the candidate-count
field before serialization. Reconstruct that ONE known field strictly. No model,
probabilities, candidates, targets, threshold rules, or stored bytes are changed.
"""
from pathlib import Path
import json
import numpy as np
import run_baseline_interval_quality_repaired as R
import optimized_baseline_crossfit as B


def producer_metadata_hash(row):
    d=dict(row)
    counts=d.get('prediction_candidate_counts')
    if not isinstance(counts,dict):raise ValueError('candidate count field must be a dictionary')
    ids=d.get('predict')
    if not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('invalid prediction identifiers')
    converted={}
    for k,v in counts.items():
        if not isinstance(k,str) or not k.isdecimal() or str(int(k))!=k or type(v) is not int or v<0:
            raise ValueError('noncanonical candidate count key/value')
        converted[int(k)]=v
    if set(converted)!=set(ids):raise ValueError('candidate counts do not cover prediction ids')
    d['prediction_candidate_counts']=converted
    return B.object_hash(d)


def load_frozen_head(name,records):
    base=R.HEAD/'fits'/name
    row=json.loads(base.with_suffix('.json').read_text('utf-8'));signature=row.pop('metadata_sha256')
    if producer_metadata_hash(row)!=signature:raise ValueError('quality producer metadata signature differs')
    row['metadata_sha256']=signature;protocol=R.check_protocol()
    if row['interval_protocol_sha256']!=protocol['receipt_sha256'] or B.digest(base.with_suffix('.npz'))!=row['npz_sha256'] or B.digest(base.with_suffix('.model.json'))!=row['model_sha256']:
        raise ValueError('quality model/cache bytes differ')
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as a:
        keys={f'{kind}_{i}' for kind in ('intervals','scores','frame','video') for i in row['predict']}
        if set(a.files)!=keys:raise ValueError('quality OOF keys differ')
        items={i:{'intervals':a[f'intervals_{i}'].copy(),'scores':a[f'scores_{i}'].copy(),
            'frame':a[f'frame_{i}'].copy(),'video':float(a[f'video_{i}'])} for i in row['predict']}
    for i,item in items.items():
        n=row['prediction_candidate_counts'][str(i)]
        if item['intervals'].shape!=(n,2) or item['intervals'].dtype!=np.dtype(np.int32) or item['scores'].shape!=(n,) or item['frame'].shape!=(records[i]['frames'],):
            raise ValueError('quality cache shape/dtype differs')
        for field in ('scores','frame'):
            p=item[field]
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid probability cache')
        if not np.isfinite(item['video']) or not 0<=item['video']<=1:raise ValueError('invalid video evidence')
        pairs=item['intervals']
        if n and (np.any(pairs[:,0]<0) or np.any(pairs[:,1]>=records[i]['frames']) or np.any(pairs[:,0]>pairs[:,1])):
            raise ValueError('invalid candidate endpoints')
    return items,row


def main():
    protocol=R.check_protocol();rows=[]
    for path in (R.HEAD/'fits').glob('*.json'):
        if path.name.endswith('.model.json'):continue
        d=json.loads(path.read_text('utf-8'));signature=d.pop('metadata_sha256')
        if producer_metadata_hash(d)!=signature:raise ValueError('original signature cannot be reproduced')
        rows.append({'file':str(path.relative_to(R.ROOT)),
                     'plain_string_key_hash_matches':B.object_hash(d)==signature,
                     'producer_integer_key_hash_matches':True,'sha256':B.digest(path)})
    evidence={'status':'frozen_before_selection','role':'strict_JSON_integer_key_restore_not_algorithm_change',
        'adapter_sha256':B.digest(Path(__file__)),'interval_protocol_sha256':protocol['receipt_sha256'],
        'original_quality_metadata_files':rows,'models_or_probabilities_retrained':False,
        'stored_metadata_or_signatures_rewritten':False}
    B.write_new(R.OUT/'selection_reader_receipt.json',evidence)
    R.load_head=load_frozen_head
    R.select_all()

if __name__=='__main__':main()