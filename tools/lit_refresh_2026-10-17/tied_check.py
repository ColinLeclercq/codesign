import numpy as np
rng = np.random.default_rng(0)
def tied(a, g, om): return np.array([[a, om+g],[a, g]])
# 1. char poly, controllability, constant similarity to controllable companion
for _ in range(3):
    a, g, om, beta = rng.normal(), rng.normal(), abs(rng.normal())+.1, rng.normal()
    M = tied(a, g, om); b = beta*np.ones((2,1))
    C = np.hstack([b, M@b])
    T = np.array([[1,-1],[0,om]])/(beta*om)
    Mc = T@M@np.linalg.inv(T); bc = T@b
    print('trace', np.trace(M), a+g, 'det', np.linalg.det(M), -a*om, 'detCtrb', np.linalg.det(C), -beta**2*om)
    print(' companion?', np.round(Mc,12).tolist(), np.round(bc.ravel(),12).tolist())
# 2. shifted RH rows linear in (a,g): det(M+eps I) = eps^2 + eps(a+g) - a om
eps=0.05
for _ in range(3):
    a, g, om = rng.normal(), rng.normal(), 1.3
    print('det(M+eps)', np.linalg.det(tied(a,g,om)+eps*np.eye(2)), eps**2+eps*(a+g)-a*om)
# 3. double root at -eps: rows tr+2eps=0, det(M+eps)=0 ; gradients wrt (a,g)
om=1.3; # solve a+g=-2eps, eps^2+eps(a+g)-a om=0 -> eps^2-2eps^2 - a om=0 -> a=-eps^2/om
a=-eps**2/om; g=-2*eps-a; M=tied(a,g,om)
print('double-root block eig', np.linalg.eigvals(M), 'rank(M+epsI)', np.linalg.matrix_rank(M+eps*np.eye(2)))
print('row gradients (a,g):', [1,1], [eps-om, eps], 'independent:', abs(np.linalg.det(np.array([[1,1],[eps-om,eps]])))>0)
# old general 2x2 form [[d1,e],[f,d2]]: det(M+eps I) gradient at M=-eps I
d1=d2=-eps; e=f=0.0
grad=[d2+eps, -f, -e, d1+eps]   # d/d(d1,e,f,d2) of (d1+eps)(d2+eps)-e f
print('old form det-row gradient at M=-eps*I:', grad)
# 4. eigenvalue sqrt-sensitivity at the double root under coefficient perturbation 1e-16
for dd in [1e-16, 1e-12]:
    lam=np.linalg.eigvals(tied(a*(1+dd), g, om)); print('perturb',dd,'eig',lam, 'max Re + eps', max(lam.real)+eps)
# 5. SISO parallel of two tied blocks: Jacobian of TF coefficients wrt params; rank drop at cross-block coalescence
def tf_coeffs(p, oms):
    # p = [a1,g1,c11,c12,a2,g2,c21,c22]; beta=1 ; returns num/den coefficients of sum, den monic degree 4
    num=np.zeros(4); den=np.array([1.0])
    blocks=[]
    for k in range(2):
        a,g,c1,c2=p[4*k:4*k+4]; M=tied(a,g,oms[k]); b=np.ones(2); c=np.array([c1,c2])
        dk=np.array([1.0, -(a+g), -a*oms[k]])
        # numerator: c adj(sI-M) b
        adj1=np.array([[1,0],[0,1]]) ; # adj(sI-M) = s I - adj... use: c (sI-M)^{-1} b = (c b s + c(adj(-M)) b)/den ; adj(sI-M)= s I + adj(-M)
        A0=np.array([[ -M[1,1], M[0,1]],[M[1,0], -M[0,0]]])  # adj(-M) = adj of -M
        nk=np.array([c@b, c@A0@b])
        blocks.append((nk,dk))
    (n1,d1),(n2,d2)=blocks
    den=np.polymul(d1,d2); num=np.polyadd(np.polymul(n1,d2), np.polymul(n2,d1))
    return np.concatenate([den[1:], np.pad(num,(4-len(num),0))])
def jac(p, oms, h=1e-7):
    f0=tf_coeffs(p,oms); J=np.zeros((len(f0),len(p)))
    for i in range(len(p)):
        q=p.copy(); q[i]+=h; J[:,i]=(tf_coeffs(q,oms)-f0)/h
    return J
oms=[1.0, 2.0]
p=np.array([-0.3,0.1,0.7,-0.2, -0.5,-0.4,0.3,0.9])
print('sv generic', np.linalg.svd(jac(p,oms),compute_uv=False))
# make block 2 denominator equal block 1's: need a2*om2 = a1*om1 and a2+g2 = a1+g1
a1,g1=p[0],p[1]; a2=a1*oms[0]/oms[1]; g2=a1+g1-a2; p2=p.copy(); p2[4]=a2; p2[5]=g2
print('sv coalesced', np.linalg.svd(jac(p2,oms),compute_uv=False))
# 6. MIMO: block with 2 inputs near derogatory M = lam I with rank-2 B (minimal), mapped to tied chart: coefficient blow-up
lam=-0.2
for d in [1e-1,1e-3,1e-6]:
    Mg=np.array([[lam, d],[0, lam+ d]]); Bg=np.eye(2); Cg=np.eye(2)
    # tied chart needs (M, B[:,0]) controllable ; similarity T s.t. T^-1 Mg T = tied, T^-1 Bg[:,0] = beta [1,1]
    b=Bg[:,0]; K=np.column_stack([b, Mg@b])
    tr=np.trace(Mg); dt=np.linalg.det(Mg); om=1.0; a=-dt/om; g=tr-a
    Mt=tied(a,g,om); bt=np.ones(2); Kt=np.column_stack([bt, Mt@bt])
    S=K@np.linalg.inv(Kt)   # Mg S = S Mt
    Bt=np.linalg.solve(S,Bg); Ct=Cg@S
    print('d',d,'cond S',np.linalg.cond(S),'|B_t col2|',np.linalg.norm(Bt[:,1]),'|C_t|',np.linalg.norm(Ct))
