import numpy as np
def tied(a, g, om): return np.array([[a, om+g],[a, g]])
lam=-0.2; om=1.0
print('MIMO 2-input block approaching derogatory M = lam*I (minimal with B = [[1,0.3],[0.2,1]], C = I)')
for d in [1e-1,1e-2,1e-3,1e-4,1e-6]:
    Mg=np.array([[lam, 0],[0, lam+d]]); Bg=np.array([[1,0.3],[0.2,1.0]]); Cg=np.eye(2)
    b=Bg[:,0]; K=np.column_stack([b, Mg@b])
    tr=np.trace(Mg); dt=np.linalg.det(Mg); a=-dt/om; g=tr-a
    Mt=tied(a,g,om); bt=np.ones(2); Kt=np.column_stack([bt, Mt@bt])
    S=K@np.linalg.inv(Kt)   # Mg S = S Mt, S bt = b
    Bt=np.linalg.solve(S,Bg); Ct=Cg@S
    assert np.allclose(Mg@S, S@Mt)
    print(f' d={d:.0e} cond(S)={np.linalg.cond(S):.2e} |Bt[:,1]|={np.linalg.norm(Bt[:,1]):.2e} |Ct|={np.linalg.norm(Ct):.2e} (a,g)=({a:.4f},{g:.4f})')
print('SISO: shared single real root between two blocks -> rank drop?')
def blk(a,g,om,c):
    M=tied(a,g,om); b=np.ones(2); A0=np.array([[-M[1,1], M[0,1]],[M[1,0], -M[0,0]]])
    return np.array([c@b, c@A0@b]), np.array([1.0, -(a+g), -a*om])
def F(p,oms):
    (n1,d1)=blk(p[0],p[1],oms[0],p[2:4]); (n2,d2)=blk(p[4],p[5],oms[1],p[6:8])
    den=np.polymul(d1,d2); num=np.polyadd(np.polymul(n1,d2), np.polymul(n2,d1))
    return np.concatenate([den[1:], np.pad(num,(4-len(num),0))])
def ag_from_roots(r1,r2,om): # char poly s^2 -(a+g)s - a om = (s-r1)(s-r2)
    tr=r1+r2; dt=r1*r2; a=-dt/om; return a, tr-a
oms=[1.0,2.0]
for shared in [False, True]:
    a1,g1=ag_from_roots(-0.05,-1.0,oms[0]); a2,g2=ag_from_roots(-0.05 if shared else -0.07,-3.0,oms[1])
    p=np.array([a1,g1,0.7,-0.2,a2,g2,0.3,0.9]); h=1e-7
    J=np.column_stack([(F(p+h*np.eye(8)[i],oms)-F(p,oms))/h for i in range(8)])
    print(' shared root' if shared else ' distinct roots', np.linalg.svd(J,compute_uv=False)[-3:])
