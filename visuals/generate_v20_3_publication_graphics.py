from __future__ import annotations
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt
from ron.experiments.scientific_protocol import ScientificProtocolConfig, run_scientific_protocol
from ron.experiments.packet_analysis import PacketAnalysisConfig, run_packet_analysis

OUT = Path("visuals/v20_3")
OUT.mkdir(parents=True, exist_ok=True)

def _read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k, v in list(rr.items()):
                if k != "variant":
                    rr[k] = float(v)
            rows.append(rr)
    return rows

def _line(rows, metric, title, ylabel, filename):
    fig = plt.figure(figsize=(9,5.5))
    for variant in ["pure", "triaxial", "hybrid"]:
        s = [r for r in rows if r["variant"] == variant]
        steps = sorted(set(int(r["step"]) for r in s))
        xs, ys = [], []
        for step in steps:
            vals = [float(r[metric]) for r in s if int(r["step"]) == step]
            xs.append(step); ys.append(sum(vals)/max(len(vals),1))
        plt.plot(xs, ys, marker="o", label=variant)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(OUT/filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _ci_bar(summary, metric, title, ylabel, filename):
    fig = plt.figure(figsize=(8,5.2))
    variants = ["pure", "triaxial", "hybrid"]
    means = [summary["metric_summary"][metric][v]["mean"] for v in variants]
    lows = [summary["metric_summary"][metric][v]["ci95"][0] for v in variants]
    highs = [summary["metric_summary"][metric][v]["ci95"][1] for v in variants]
    yerr = [[m-l for m,l in zip(means,lows)], [h-m for m,h in zip(means,highs)]]
    plt.bar(variants, means, yerr=yerr, capsize=5)
    plt.title(title)
    plt.ylabel(ylabel)
    fig.savefig(OUT/filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _effect_plot(summary):
    keys = list(summary["comparisons"].keys())
    vals = [summary["comparisons"][k]["mean_diff"] for k in keys]
    fig = plt.figure(figsize=(10,5.5))
    plt.bar(range(len(keys)), vals)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(len(keys)), keys, rotation=35, ha="right")
    plt.title("RON V20.3 — effets vs RON pur")
    plt.ylabel("différence moyenne")
    fig.savefig(OUT/"v20_3_effects_vs_pure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def _confusion(packet_report):
    cm = packet_report["confusion_proxy"]
    fig = plt.figure(figsize=(6.5,5.5))
    plt.imshow(cm)
    plt.title("RONPacket — confusion proxy action")
    plt.xlabel("prédit")
    plt.ylabel("cible")
    faces = ["+X","-X","+Y","-Y","+Z","-Z"]
    plt.xticks(range(6), faces)
    plt.yticks(range(6), faces)
    for i,row in enumerate(cm):
        for j,v in enumerate(row):
            plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar()
    fig.savefig(OUT/"v20_3_packet_confusion_proxy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    summary = run_scientific_protocol(ScientificProtocolConfig(steps=2, batch_size=5, seeds=1, out_dir="runs/v20_3_publication_protocol"))
    packet = run_packet_analysis(PacketAnalysisConfig(samples=12, out_dir="runs/v20_3_publication_packet"))
    rows = _read_csv(summary["all_steps_csv"])
    _line(rows, "total_loss", "RON V20.3 — perte moyenne multi-graines", "loss", "v20_3_loss_mean_by_step.png")
    _line(rows, "action_acc", "RON V20.3 — action accuracy multi-graines", "accuracy", "v20_3_action_accuracy_mean_by_step.png")
    _line(rows, "release", "RON V20.3 — release multi-graines", "release", "v20_3_release_mean_by_step.png")
    _line(rows, "residual", "RON V20.3 — residual multi-graines", "residual", "v20_3_residual_mean_by_step.png")
    _ci_bar(summary, "total_loss", "RON V20.3 — final loss avec CI bootstrap", "loss", "v20_3_final_loss_ci.png")
    _ci_bar(summary, "action_acc", "RON V20.3 — final action accuracy avec CI bootstrap", "accuracy", "v20_3_final_action_acc_ci.png")
    _effect_plot(summary)
    _confusion(packet)
    report = {"protocol": summary, "packet": packet, "generated": sorted(p.name for p in OUT.glob("*.png"))}
    (OUT/"v20_3_publication_visual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "generated": report["generated"]}, indent=2))

if __name__ == "__main__":
    main()
