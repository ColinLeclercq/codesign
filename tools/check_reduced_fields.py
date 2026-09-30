#!/usr/bin/env python3
"""check_reduced_fields.py -- reduced-space field gradients (Gabor atoms,
projected Jacobians, STAGE6 6.29) against the lifted path, on the at_dec record."""
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import numpy as np
from check_mimo import load
from reduced import ReducedOracle
from packer import PackerMIMO
import fieldbasis as FB
rec=sys.argv[1] if len(sys.argv) > 1 else 'results_mimo_prod/at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz'
z,D,b,c,arow,Cc,Dc=load(rec)
na_=4; pa_=np.asarray(z['atoms_a_p'],float); ps_=np.asarray(z['atoms_s_p'],float)
D.atoms_a=dict(n=na_,smin=1.0,kmax=0.7071,p0=list(pa_)); D.atoms_s=dict(n=na_,smin=1.0,kmax=0.7071,p0=list(ps_))
flags=dict(K=True,B=False,C=False,FB=True,FC=True,m=2,p=2,struct='central',loop_orders=[4,4])
va=[FB.atom_field(D.xn,*FB.split_atoms(q,na_),1.0,0.7071) for q in pa_]; vs=[FB.atom_field(D.xn,*FB.split_atoms(q,na_),1.0,0.7071) for q in ps_]
pk=PackerMIMO(D,flags,arow,Cc,Dc,np.zeros(2),np.zeros(2),va,vs)
zz=pk.z0(); arow2,nvec2,K2,xa,xs,va2,vs2=pk.unpack(zz)
bb,cc,pa,ps,nna,nns=D.bc(xa,xs,va2,vs2)
ro=ReducedOracle(D); ro.galerkin=True; clf=D.cl(bb,cc)
for w in [-2.0,-0.7,0.3,1.9,5.0]: ro.enrich(clf,w,arow2,nvec2,K2)
cl=ro.make_cl(bb,cc,D.r,D.mc); clg=cl.proxy; clg.pk=pk
mx=0
for w in np.linspace(-6,6,13):
    for P in cl.gain_k(float(w),arow2,nvec2,K2,k=2):
        x,zv=P['x'],P['z']; xf,zf=clg.lift(x,zv)
        gr=pk.grad(D.grads(clg,x,zv,arow2,nvec2,K2,xa,xs,pa,ps,nna,nns,flags))
        gf=pk.grad(D.grads(clg,xf,zf,arow2,nvec2,K2,xa,xs,pa,ps,nna,nns,flags,_full=True))
        mx=max(mx,np.linalg.norm(gr-gf)/np.linalg.norm(gf))
print('atoms: packed gradient reduced vs lifted, max rel', f'{mx:.1e}', 'len', len(gr), 'fimg built', clg._fimg is not False and clg._fimg is not None)
