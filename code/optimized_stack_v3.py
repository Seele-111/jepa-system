"""Identity-shrunk monotone OOF ensemble; NumPy-only, not calibrated risk."""
from __future__ import annotations
from collections.abc import Mapping
from itertools import product
import numpy as np

KIND='monotone_stack_logit_v3'
EPS=1e-5
BRANCHES=((('corrected_motion_rf',.5),('corrected_motion_et',.5)),(('local_corrected_motion_et',.5),('noglobal_corrected_motion_et',.5)),(('rgb_motion_tcn',1.),))
FAST_BRANCHES=((('motion_rf',1.),),(('motion_et',1.),),(('local_motion_et',1.),))
ANCHOR=np.array([1/3,1/3,1/3,0.,0.])
LOWER=np.array([0.,0.,0.,0.,-3.]);UPPER=np.array([3.,3.,3.,3.,3.])


def _array(value,name,*,probabilities=False):
    try:
        raw=np.asarray(value)
        if np.iscomplexobj(raw):raise ValueError('complex')
        result=np.asarray(raw,np.float64)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('invalid stack '+name) from exc
    if not np.isfinite(result).all():raise ValueError('nonfinite stack '+name)
    if probabilities and np.any((result<0)|(result>1)):raise ValueError('stack '+name+' outside [0,1]')
    return result


def _branch_schema(branches):
    if isinstance(branches,(str,bytes,Mapping)):raise ValueError('invalid stack branch schema')
    try:branches=tuple(tuple((str(r),float(w)) for r,w in b) for b in branches)
    except (TypeError,ValueError) as exc:raise ValueError('invalid stack branch schema') from exc
    if len(branches)!=3:raise ValueError('stack requires exactly three branches')
    for branch in branches:
        if not branch or len({r for r,w in branch})!=len(branch):raise ValueError('invalid stack branch members')
        weights=np.asarray([w for r,w in branch])
        if any(not r for r,w in branch) or not np.isfinite(weights).all() or np.any(weights<=0) or abs(weights.sum()-1)>1e-10:raise ValueError('invalid stack branch weights')
    if branches not in (BRANCHES,FAST_BRANCHES):raise ValueError('unknown stack branch profile')
    return branches


def validate_state(state,available_recipes=None):
    if not isinstance(state,Mapping) or state.get('kind')!=KIND:raise ValueError('invalid stack state kind')
    required={'kind','branches','coefficients','ridge'}
    allowed=required|{'weighting','fit_role','independent_probability_calibration','business_risk_probability','initial_loss','final_loss','iterations'}
    if not required<=set(state) or set(state)-allowed:raise ValueError('incomplete or unknown stack state fields')
    if isinstance(state['ridge'],(bool,np.bool_)):raise ValueError('invalid stack ridge type')
    if not isinstance(state['coefficients'],(list,tuple,np.ndarray)):raise ValueError('invalid stack coefficients structure')
    if np.ndim(state['coefficients'])!=1:raise ValueError('invalid stack coefficient shape')
    if any(isinstance(v,(bool,np.bool_)) for v in state['coefficients']):raise ValueError('invalid stack coefficient type')
    for key in ['initial_loss','final_loss']:
        if key in state and (_array(state[key],key).ndim!=0 or float(state[key])<0):raise ValueError('invalid stack loss metadata')
    for key in ['independent_probability_calibration','business_risk_probability']:
        if key in state and not isinstance(state[key],(bool,np.bool_)):raise ValueError('invalid stack probability metadata')
    for key in ['weighting','fit_role']:
        if key in state and not isinstance(state[key],str):raise ValueError('invalid stack role metadata')
    if 'iterations' in state and (isinstance(state['iterations'],bool) or not isinstance(state['iterations'],(int,np.integer)) or not 0<=state['iterations']<=60):raise ValueError('invalid stack iterations')
    branches=_branch_schema(state['branches']);beta=_array(state['coefficients'],'coefficients')
    if beta.shape!=(5,) or np.any(beta<LOWER) or np.any(beta>UPPER):raise ValueError('invalid stack coefficients')
    ridge=_array(state['ridge'],'ridge')
    if ridge.ndim!=0 or float(ridge)!=1.:raise ValueError('stack ridge fixed at 1')
    if available_recipes is not None and {r for b in branches for r,w in b}!=set(available_recipes):raise ValueError('stack recipe schema mismatch')
    return branches,beta


def _logit(p):
    p=np.clip(p,EPS,1-EPS);return np.log(p)-np.log1p(-p)


def _features(frame,video,branches):
    if not isinstance(frame,Mapping) or not isinstance(video,Mapping):raise ValueError('stack needs member predictions')
    branches=_branch_schema(branches);columns=[];vsum=0.;n=None
    for branch in branches:
        bp=None;bv=0.
        for recipe,weight in branch:
            if recipe not in frame or recipe not in video:raise ValueError('missing stack member')
            p=_array(frame[recipe],'frame',probabilities=True);v=_array(video[recipe],'video',probabilities=True)
            if p.ndim!=1 or v.ndim!=0:raise ValueError('invalid stack probability shape')
            if n is None:n=len(p)
            if len(p)!=n:raise ValueError('unaligned stack members')
            bp=weight*p if bp is None else bp+weight*p;bv+=weight*float(v)
        columns.append(_logit(bp));vsum+=bv/3
    columns.extend([np.full(n,_logit(np.asarray(vsum))),np.ones(n)])
    return np.stack(columns,axis=1),float(vsum)


def apply_stack(frame_by_recipe,video_by_recipe,state):
    branches,beta=validate_state(state,set(frame_by_recipe))
    if set(video_by_recipe)!=set(frame_by_recipe):raise ValueError('stack video schema mismatch')
    x,v=_features(frame_by_recipe,video_by_recipe,branches);z=x@beta
    return (1/(1+np.exp(-np.clip(z,-700,700)))).astype(np.float32),v


def _direction(beta,gradient,hessian):
    lower,upper=LOWER-beta,UPPER-beta;best=np.zeros(5);value=0.
    for status in product((-1,0,1),repeat=5):
        status=np.asarray(status);free=np.flatnonzero(status==0);fixed=np.flatnonzero(status!=0);d=np.zeros(5)
        d[status==-1]=lower[status==-1];d[status==1]=upper[status==1]
        if len(free):d[free]=np.linalg.solve(hessian[np.ix_(free,free)],-gradient[free]-hessian[np.ix_(free,fixed)]@d[fixed])
        if np.any(d<lower-1e-12) or np.any(d>upper+1e-12):continue
        d=np.clip(d,lower,upper);cost=float(gradient@d+.5*d@hessian@d)
        if cost<value:best=d;value=cost
    return best


def fit_stack(frame_list,video_list,labels_list,branches=BRANCHES,content_mass=None):
    """SUM equal-content per-video mean BCE + fixed ridge=1 around equal logits.

    Inputs must be outer-training inner OOF. No class rebalancing. Aliases get
    mass=1/alias_count. This cross-meta fitting retains base OOF dependencies.
    """
    branches=_branch_schema(branches)
    if any(isinstance(v,(str,bytes,Mapping)) for v in (frame_list,video_list,labels_list)):raise ValueError('stack fitting expects sequences')
    frame_list,video_list,labels_list=list(frame_list),list(video_list),list(labels_list);n=len(labels_list)
    if not n or len(frame_list)!=n or len(video_list)!=n:raise ValueError('unaligned stack fitting videos')
    mass=np.ones(n) if content_mass is None else _array(content_mass,'content weights')
    if mass.shape!=(n,) or np.any(mass<=0):raise ValueError('invalid stack content weights')
    xs=[];ys=[];ws=[]
    for frame,video,labels,m in zip(frame_list,video_list,labels_list,mass):
        x,_=_features(frame,video,branches);y=_array(labels,'labels')
        if not len(x) or y.shape!=(len(x),) or np.any((y!=0)&(y!=1)):raise ValueError('invalid stack labels')
        xs.append(x);ys.append(y);ws.append(np.full(len(x),m/len(x)))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);beta=ANCHOR.copy()
    def loss(b):
        z=x@b;return float(np.sum(w*(np.logaddexp(0,z)-y*z))+.5*np.sum((b-ANCHOR)**2))
    initial=loss(beta);iterations=0
    if len(np.unique(y))>1:
        for iteration in range(60):
            z=x@beta;p=1/(1+np.exp(-np.clip(z,-700,700)));gradient=x.T@(w*(p-y))+(beta-ANCHOR)
            hessian=x.T@((w*p*(1-p))[:,None]*x)+np.eye(5);d=_direction(beta,gradient,hessian)
            if np.max(np.abs(d))<1e-9:break
            scale=1.;old=loss(beta)
            while scale>=2**-24:
                candidate=np.clip(beta+scale*d,LOWER,UPPER)
                if loss(candidate)<=old+1e-4*scale*float(gradient@d):break
                scale*=.5
            if scale<2**-24:break
            beta=candidate;iterations=iteration+1
    state={'kind':KIND,'branches':[[[r,w] for r,w in b] for b in branches],'coefficients':beta.tolist(),'ridge':1.,
           'weighting':'inverse_video_length_inverse_content_alias_count','fit_role':'outer_training_inner_OOF_only',
           'independent_probability_calibration':False,'business_risk_probability':False,
           'initial_loss':initial,'final_loss':loss(beta),'iterations':iterations}
    validate_state(state);return state
