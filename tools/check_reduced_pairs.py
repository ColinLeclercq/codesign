#!/usr/bin/env python3
"""check_reduced_pairs.py -- the reduced-space rows against their lifted
equivalents (STAGE6 6.29): gradients (controller, placement), gprime and the
singular-vector scaling, batched vs per-row sigma, and the Gram-matrix
full-model residuals of the near-line Ritz pairs against direct products.
    python tools/check_reduced_pairs.py results_mimo_prod/kbc_cen5/K_B_C_s0_ne100_mimo2x2_central.npz
The atoms (field) gradients: tools/check_reduced_fields.py."""
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import numpy as np
from check_mimo import load
from reduced import ReducedOracle
rec = sys.argv[1]
z,D,b,c,arow,Cc,Dc=load(rec)
ro=ReducedOracle(D); ro.galerkin=True
clf=D.cl(b,c)
for w in [-2.0,-0.7,0.3,1.9,5.0]:
    ro.enrich(clf, w, arow, Cc, Dc)
cl=ro.make_cl(b,c,D.r,D.mc); clg=cl.proxy
kd=ro.V.shape[1]; print('kd',kd, 'orth', np.linalg.norm(ro.V.conj().T@ro.V-np.eye(kd)))
flags=dict(K=True,B=True,C=True,FB=False,FC=False)
xa=np.asarray(z['xa'],float); xs=np.asarray(z['xs'],float)
mx=0; mg=0
for w in np.linspace(-6,6,25):
    for P in cl.gain_k(float(w), arow, Cc, Dc, k=2):
        x,zz=P['x'],P['z']; xf,zf=clg.lift(x,zz)
        gr=D.grads(clg,x,zz,arow,Cc,Dc,xa,xs,None,None,None,None,flags)
        gf=D.grads(clg,xf,zf,arow,Cc,Dc,xa,xs,None,None,None,None,flags,_full=True)
        for k_ in gr:
            a_,b_=np.asarray(gr[k_]).ravel(),np.asarray(gf[k_]).ravel()
            mx=max(mx,np.linalg.norm(a_-b_)/max(np.linalg.norm(b_),1e-300))
        gp_old=float(np.imag(np.vdot(zf[:D.n], D.M@xf[:D.n])+np.vdot(zf[D.n:],xf[D.n:])))
        mg=max(mg,abs(gp_old-P['gprime'])/max(abs(gp_old),1e-12))
        # scaling: x_p^H M x_p = sigma^2
        q=np.vdot(xf[:D.n],D.M@xf[:D.n]).real
print('grads reduced vs lifted: max rel', f'{mx:.1e}', ' gprime', f'{mg:.1e}', ' last qMq/sigma^2', q/P['sigma']**2)
# batch pairs equal to gain_k pairs
ws=np.linspace(-6,6,25)
B=cl.gain_batch(ws, arow, Cc, Dc, np.ones(25,bool), par=False)
d=max(abs(B[i]['sigma']-cl.gain(float(w),arow,Cc,Dc)['sigma'])/B[i]['sigma'] for i,w in enumerate(ws))
print('batch vs per-row sigma', f'{d:.1e}')
# residuals: Gram vs direct
pairs=cl.near_line_pairs(arow, Cc, Dc, 0.05, reach=3.0, res_tol=1.0, mc=D.mc, kmax=8)
Acl,Mcl=cl._red_pencil
from scipy.linalg import eig
lam,VL,VR=eig(Acl,Mcl,left=True,right=True)
mr=0
for j in np.argsort(-lam.real)[:10]:
    y=VR[:,j]; l=lam[j]
    rg=cl._full_residual(y,complex(l),arow,Cc,Dc,D.mc)
    vf=np.concatenate([ro.V@y[:kd], y[kd:]]); Av=clf.Acl_apply(vf,arow,Cc,Dc); Mv=clf.Mcl_apply(vf)
    rd=np.linalg.norm(Av-l*Mv)/max(abs(l)*np.linalg.norm(Mv)+np.linalg.norm(Av),1e-300)
    print(f'  lam {l:.4f}: residual Gram {rg:.3e} direct {rd:.3e}')
print('pairs', len(pairs))
