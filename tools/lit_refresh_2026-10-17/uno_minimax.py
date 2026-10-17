import re,sys,statistics as st
txt=open(sys.argv[1]).read().splitlines()[1838:2210]
names="polak1 polak2 polak3 polak4 polak5 polak6 rosenmmx minmaxbd minmaxrb madsen makela1 makela2 makela3 mifflin1 mifflin2 gigomez1 chaconn1 chaconn2 cb2 cb3 demymalo womflet kiwcresc".split()
cols="uno_filtersqp uno_ipopt filterSQP IPOPT SNOPT MINOS LANCELOT LOQO CONOPT".split()
rows={}
for l in txt:
    t=l.split()
    if t and t[0] in names and len(t)>=10: rows[t[0]]=t[1:10]
print(len(rows),"rows")
for j,c in enumerate(cols):
    v=[int(r[j]) for r in rows.values() if r[j].isdigit()]; f=sum(1 for r in rows.values() if not r[j].isdigit())
    print(f"{c:14s} solved {len(v):2d} fail {f} median {st.median(v):6.1f} max {max(v)}")
