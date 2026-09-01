"""Aggregate benchmark JSON files and generate Pareto figures."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--out",default="figures")
    a=p.parse_args(); rows=[]
    for f in Path(a.root).glob("**/metrics.json"):
        d=json.loads(f.read_text())
        if d["config"].get("synthetic"): continue
        rows.append({"model":d["config"]["model"],"seed":d["config"]["seed"],"accuracy":d.get("test",d["best_val"])["accuracy"],"parameters":d["model"]["parameters"],"memory_mib":d["model"]["resident_model_bytes"]/2**20,"latency_ms":d["profile_batch1"]["latency_ms"]})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(rows,indent=2))
    if not rows: return
    with (out/"summary.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for x,name,label in [("parameters","accuracy_vs_parameters","Trainable parameters"),("memory_mib","accuracy_vs_memory","Resident model memory (MiB)"),("latency_ms","accuracy_vs_latency","Batch-1 latency (ms)")]:
        fig,ax=plt.subplots()
        for m in sorted(set(r["model"] for r in rows)):
            rr=[r for r in rows if r["model"]==m]; groups={}
            for r in rr: groups.setdefault(r[x],[]).append(r["accuracy"])
            xs=sorted(groups); ys=np.array([np.mean(groups[z]) for z in xs]); sd=np.array([np.std(groups[z]) for z in xs])
            ax.plot(xs,ys,marker="o",label=m); ax.fill_between(xs,ys-sd,ys+sd,alpha=.2)
        ax.set_xlabel(label); ax.set_ylabel("CIFAR-10 test accuracy"); ax.legend(); fig.savefig(out/f"{name}.png"); fig.savefig(out/f"{name}.pdf"); plt.close(fig)
if __name__=="__main__": main()
