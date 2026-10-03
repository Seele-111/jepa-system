import unittest
from copy import deepcopy
import numpy as np
from optimized_stack_v3 import BRANCHES,ANCHOR,apply_stack,fit_stack,validate_state

class StackTests(unittest.TestCase):
    def samples(self):
        rng=np.random.default_rng(220);frames=[];videos=[];labels=[];recipes={r for b in BRANCHES for r,w in b}
        for i in range(12):
            y=np.zeros(40);y[8:20]=i%2
            frames.append({r:np.clip(.1+.6*y+rng.normal(0,.12,40),.001,.999) for r in sorted(recipes)})
            videos.append({r:.25+.5*(i%2) for r in recipes});labels.append(y)
        return frames,videos,labels
    def test_fit_descends_deterministic_portable(self):
        f,v,y=self.samples();s=fit_stack(f,v,y);s2=fit_stack(f,v,y)
        self.assertLess(s['final_loss'],s['initial_loss']);self.assertEqual(s,s2)
        p,video=apply_stack(f[0],v[0],s);self.assertEqual(p.shape,(40,));self.assertTrue(np.all((p>=0)&(p<=1)));self.assertAlmostEqual(video,.25)
    def test_monotone_each_member_and_video(self):
        f,v,y=self.samples();s=fit_stack(f,v,y);p,_=apply_stack(f[0],v[0],s)
        for r in f[0]:
            ff=deepcopy(f[0]);ff[r]=np.minimum(ff[r]+.1,1);q,_=apply_stack(ff,v[0],s);self.assertTrue(np.all(q>=p-1e-7))
            vv=dict(v[0]);vv[r]=min(1,vv[r]+.2);q,_=apply_stack(f[0],vv,s);self.assertTrue(np.all(q>=p-1e-7))
    def test_single_class_fallback_preserves_anchor(self):
        f,v,y=self.samples();s=fit_stack(f,v,[np.zeros(40)]*len(y));np.testing.assert_array_equal(s['coefficients'],ANCHOR)
    def test_content_alias_weight_prevents_replication_changing_fit(self):
        f,v,y=self.samples();s=fit_stack(f,v,y);d=fit_stack(f+f,v+v,y+y,content_mass=np.full(2*len(y),.5))
        np.testing.assert_allclose(s['coefficients'],d['coefficients'],rtol=1e-8,atol=1e-8)
    def test_reject_corrupt_state(self):
        f,v,y=self.samples();s=fit_stack(f,v,y)
        for k,value in [('kind','typo'),('coefficients',[0,0,0,0]),('coefficients',[-1,1,1,1,0]),('coefficients',[1,1,1,1,np.nan]),('ridge',.5),('branches',[]),('ridge',True),('coefficients',1),('coefficients',[True,1,1,1,0]),('iterations',1.5),('unknown_field',0),('final_loss',float('nan'))]:
            with self.subTest(k=k,value=value),self.assertRaises(ValueError):
                ss=deepcopy(s);ss[k]=value;validate_state(ss)
    def test_reject_invalid_fit_inputs(self):
        f,v,y=self.samples()
        for yy in [y[:-1],[np.ones(3)]*len(y),[np.full(40,.5)]*len(y),[np.full(40,np.nan)]*len(y)]:
            with self.assertRaises(ValueError):fit_stack(f,v,yy)
        with self.assertRaises(ValueError):fit_stack(f,v,y,content_mass=np.zeros(len(y)))
    def test_reject_missing_unaligned_or_invalid_member(self):
        f,v,y=self.samples();s=fit_stack(f,v,y);r=next(iter(f[0]))
        for value in [np.ones(1),np.full(40,1.1),np.full(40,np.nan)]:
            ff=dict(f[0]);ff[r]=value
            with self.assertRaises(ValueError):apply_stack(ff,v[0],s)
        ff=dict(f[0]);ff.pop(r)
        with self.assertRaises(ValueError):apply_stack(ff,v[0],s)

if __name__=='__main__':unittest.main()
