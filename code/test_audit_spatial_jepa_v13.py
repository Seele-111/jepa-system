"""Auditor tamper tests without estimator fitting or real cache mutation."""
import copy,io,json,unittest
import numpy as np
from unittest.mock import patch
import audit_spatial_jepa_v13 as V13
import audit_feature_blocks_v10 as A
from test_spatial_jepa_v13 import spatial_record
from optimized_spatial_jepa_v13 import feature_view_spatial


class FakeEvidence:
    def __init__(self,row,data):self.row=row;self.data=data
    def json(self,path):return self.row
    def read(self,path,expected=None):
        if expected is not None:A.require(A.sha(self.data)==expected,'test NPZ hash mismatch')
        return self.data


class SpatialAudit(unittest.TestCase):
    def fixture(self):
        records=[spatial_record(i) for i in range(9)]
        for r in records:r.update(event_count=1,generator='synthetic')
        ids=list(range(9));parts=A.expected_partitions(records,ids,5)
        names=feature_view_spatial(V13.RECIPE,records[0])[2]
        h=A.value_hash({'feature_view':'spatial-jepa-v13','pca':None,'frame_feature_names':names})
        proof=[{'partition':k,'fit_content_sha256':sorted(records[i]['sha256'] for i in p['fit']),'transform_sha256':h} for k,p in enumerate(parts)]
        arrays={f'inner_{head}_{i}':(np.full(60,.5,np.float32) if head=='frame' else np.asarray(.7)) for i in ids for head in ['frame','video']}
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue()
        row={'scope':5,'final_selection':True,'recipe':V13.RECIPE,'train':ids,'validation':[],
            'inner_partitions':parts,'training_receipt_sha256':'receipt-test','outer_seed':None,
            'fit_evidence':proof,'npz_sha256':A.sha(data)}
        row['signature']=A.value_hash(row)
        return records,row,data,arrays

    def resign(self,row):row['signature']=A.value_hash({k:v for k,v in row.items() if k!='signature'})

    def test_valid_direct_inner_partition_and_probabilities(self):
        records,row,data,_=self.fixture();inner,outer,_=V13.verify_spatial_cache(FakeEvidence(row,data),5,records,[],{'receipt_sha256':'receipt-test'},final=True)
        self.assertEqual(set(inner.frame),set(range(9)));self.assertFalse(outer.frame)

    def test_forged_fit_content_or_transform_rejected(self):
        records,row,data,_=self.fixture()
        for field in ['fit_content_sha256','transform_sha256']:
            mutated=copy.deepcopy(row);mutated['fit_evidence'][0][field]=[] if field=='fit_content_sha256' else '0'*64;self.resign(mutated)
            with self.assertRaises(A.AuditError):V13.verify_spatial_cache(FakeEvidence(mutated,data),5,records,[],{'receipt_sha256':'receipt-test'},final=True)

    def test_probability_range_or_key_coverage_rejected(self):
        records,row,data,arrays=self.fixture()
        for kind in ['range','coverage']:
            a=copy.deepcopy(arrays);r=copy.deepcopy(row)
            if kind=='range':a['inner_frame_0'][0]=1.1
            else:a.pop('inner_video_0')
            stream=io.BytesIO();np.savez_compressed(stream,**a);d=stream.getvalue();r['npz_sha256']=A.sha(d);self.resign(r)
            with self.assertRaises(A.AuditError):V13.verify_spatial_cache(FakeEvidence(r,d),5,records,[],{'receipt_sha256':'receipt-test'},final=True)

    def test_signature_and_partition_tamper_rejected(self):
        records,row,data,_=self.fixture();r=copy.deepcopy(row);r['scope']=4
        with self.assertRaises(A.AuditError):V13.verify_spatial_cache(FakeEvidence(r,data),5,records,[],{'receipt_sha256':'receipt-test'},final=True)
        self.resign(r)
        with self.assertRaises(A.AuditError):V13.verify_spatial_cache(FakeEvidence(r,data),5,records,[],{'receipt_sha256':'receipt-test'},final=True)

if __name__=='__main__':unittest.main()
