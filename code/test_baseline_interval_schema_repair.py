"""Integrate the actual interval feature tuple with serialized protocol and fit/cache flow."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import optimized_interval_quality as Q
import run_baseline_interval_quality_repaired as R


class SchemaRoundtripIntegration(unittest.TestCase):
    def test_actual_head_fit_accepts_json_restored_feature_names(self):
        records=[{'sha256':str(i),'fps':24.,'frames':4,'labels':np.array([0,1,1,0] if i%2==0 else [0,0,0,0],np.uint8)} for i in range(4)]
        raw={i:np.arange(4,dtype=np.float32)[:,None] for i in range(4)}
        p={i:np.full(4,.6,np.float32) for i in range(4)};v=dict.fromkeys(range(4),.7)
        _,names=Q.interval_features(p[0],.7,24.,[(0,3)],raw[0])
        protocol=json.loads(json.dumps({'receipt_sha256':'head','interval_feature_names':names}))
        self.assertIsInstance(protocol['interval_feature_names'],list)
        fitted=[]
        class FixedHead:
            def predict(self,x):return np.full(len(x),.5)
        def capture_fit(x,y,w,seed):
            self.assertEqual(x.ndim,2);self.assertEqual(x.shape[1],len(names))
            self.assertEqual(y.shape,w.shape);self.assertEqual(len(x),len(y));self.assertTrue(np.all(w>=0))
            fitted.append((x.shape,len(y)));return FixedHead()
        train=R.Probabilities({i:p[i] for i in (0,1)},{i:v[i] for i in (0,1)},None)
        pred=R.Probabilities({i:p[i] for i in (2,3)},{i:v[i] for i in (2,3)},None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(R,'HEAD',Path(tmp)),\
             patch.object(R,'STATE',({},records,{},raw,['evidence'])),\
             patch.object(R,'check_protocol',return_value=protocol),\
             patch.object(R.B,'check_protocol',return_value={'receipt_sha256':'base'}),\
             patch.object(R,'head_inputs',return_value=(train,pred,[0,1],[2,3],['deep'],['inner'])),\
             patch.object(Q,'fit_quality_head',side_effect=capture_fit),\
             patch.object(Q,'export_quality_head',return_value={'test_only':True}),\
             patch.object(Q,'portable_predict_quality',side_effect=lambda m,x:np.full(len(x),.5,np.float32)):
            folder=Path(tmp)/'fits';folder.mkdir()
            result=R.fit_head(0,0)
            row=json.loads((folder/'s0_inner0.json').read_text())
            model=json.loads((folder/'s0_inner0.model.json').read_text())
            self.assertEqual(model['interval_feature_names'],list(names))
            self.assertEqual(row['fit'],[0,1]);self.assertEqual(row['predict'],[2,3])
            self.assertEqual(row['portable_max_error'],0);self.assertEqual(result['name'],'s0_inner0')
        self.assertEqual(len(fitted),1)

    def test_wrong_feature_schema_still_rejected_before_fit(self):
        records=[{'sha256':str(i),'fps':24.,'frames':4,'labels':np.ones(4,np.uint8)} for i in range(2)]
        p=R.Probabilities({0:np.full(4,.6,np.float32)},{0:.7},None)
        pred=R.Probabilities({1:np.full(4,.6,np.float32)},{1:.7},None)
        raw={i:np.zeros((4,1),np.float32) for i in range(2)}
        with tempfile.TemporaryDirectory() as tmp,patch.object(R,'HEAD',Path(tmp)),\
             patch.object(R,'STATE',({},records,{},raw,['evidence'])),\
             patch.object(R,'check_protocol',return_value={'receipt_sha256':'bad','interval_feature_names':['wrong']}),\
             patch.object(R,'head_inputs',return_value=(p,pred,[0],[1],[],[])),\
             patch.object(Q,'fit_quality_head') as fit:
            (Path(tmp)/'fits').mkdir()
            with self.assertRaisesRegex(ValueError,'feature names differ'):R.fit_head(0,0)
            fit.assert_not_called()
            self.assertEqual(list((Path(tmp)/'fits').iterdir()),[])


if __name__=='__main__':unittest.main()