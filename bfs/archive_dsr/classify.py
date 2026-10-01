"""Branch identification: A (shear layer) vs B (free stream) from where the forcing/response energy sits."""
import numpy as np, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "fx"
m = np.load(f"modes_{tag}.npz"); G = np.loadtxt(f"gains_{tag}.txt")
Xf, Xr = m["forc_xy"], m["resp_xy"]
shear_f = (Xf[:, 0] < 4) & (np.abs(Xf[:, 1] - 1.0) < 0.4)          # around separation
low_r = Xr[:, 1] < 1.2                                               # shear layer / bubble height
for row in G:
    om = row[0]; line = []
    for k in range(len(row) - 1):
        fu, fv = m[f"w{om}_k{k}_forc_u"], m[f"w{om}_k{k}_forc_v"]
        ru, rv = m[f"w{om}_k{k}_resp_u"], m[f"w{om}_k{k}_resp_v"]
        ef = np.abs(fu)**2 + np.abs(fv)**2; er = np.abs(ru)**2 + np.abs(rv)**2
        a, b = ef[shear_f].sum() / ef.sum(), er[low_r].sum() / er.sum()
        line.append(f"{row[k+1]:9.3g} [{a:.2f}|{b:.2f}]")
    print(f"w={om:4.2f}: " + "  ".join(line[:8]))
