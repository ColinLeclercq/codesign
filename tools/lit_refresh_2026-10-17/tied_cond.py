import numpy as np
def tied(a,g,om): return np.array([[a,om+g],[a,g]])
def from_roots(r1,r2,om):
    tr=(r1+r2).real; dt=(r1*r2).real; a=-dt/om; return tied(a,tr-a,om)
cases={'lightly damped -0.05+-2.29i':(-0.05+2.293j,-0.05-2.293j),'real pair -0.05,-1':(-0.05,-1.0),'real pair -0.05,-400':(-0.05,-400.0),'near-double real -0.05,-0.0501':(-0.05,-0.0501)}
for name,(r1,r2) in cases.items():
    dt=(r1*r2).real
    for om in [np.sqrt(abs(dt)), 1.0, abs(r1)+abs(r2)]:
        M=from_roots(r1,r2,om); w,V=np.linalg.eig(M); V=V/np.linalg.norm(V,axis=0)
        print(f'{name:32s} om={om:9.4g} cond(V)={np.linalg.cond(V):9.3g} |M|max={abs(M).max():9.3g}')
