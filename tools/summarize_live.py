"""Summarise a live ROS 2 run (adaptmap_node stats JSONL) into a section of docs/RESULTS.md.

Usage: python tools/summarize_live.py data/demo_*/stats.jsonl
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = "\n## Live ROS 2 + Gazebo run"


def main():
    path = Path(sys.argv[1])
    r = [json.loads(l) for l in path.open()][3:]  # skip warm-up frames
    span = r[-1]["stamp"] - r[0]["stamp"]
    pct = lambda v: sorted(v)[int(0.95 * len(v))]  # noqa: E731
    rows = []
    for k, name in (("preprocess", "Preprocess"), ("segment", "Segmentation"), ("fuse", "Foveated fusion"), ("total", "**Pipeline total**")):
        v = [x["timing_ms"][k] for x in r]
        rows.append(f"| {name} | {st.mean(v):.1f} | {pct(v):.1f} |")
    pub = [x["timing_ms"]["publish"] for x in r if "publish" in x["timing_ms"]]
    occ = [sum(x["occupied"].values()) for x in r]
    md = [HEAD.strip(), "",
          f"Scripted lap through the Gazebo world (`tools/capture_demo.sh`): Gazebo, RViz, the video renderer and "
          f"AdaptMap all on one RTX 4050 laptop. {len(r)} scans processed over {span:.0f} s of sim time "
          f"(**{len(r)/span:.1f} scans/s**; the node always takes the newest scan, LiDAR runs at ~8.5 Hz).", "",
          "| Stage | Mean (ms) | p95 (ms) |", "|---|---|---|", *rows,
          f"| ROS publishing (every 3rd scan) | {st.mean(pub):.1f} | {pct(pub):.1f} |", "",
          f"Peak GPU memory {max(x['gpu_mem_mb'] for x in r):.0f} MB. Points per scan after preprocessing "
          f"{st.mean(x['points']['out'] for x in r):,.0f}. Observed map cells at end of lap {occ[-1]:,} "
          f"(of {r[-1]['total_cells']:,} foveated cells; a uniform 5 cm map would hold {r[-1]['uniform_cells']:,}).", "",
          "Segmentation is slower live than offline (75 ms) because the GPU is shared with Gazebo's LiDAR "
          "rendering and the GUIs.", "",
          "Demo video: [figures/demo.mp4](figures/demo.mp4) — left: semantic classes per zone, right: elevation.", "",
          "![live](figures/demo_frame.png)", ""]
    res = ROOT / "docs" / "RESULTS.md"
    txt = res.read_text()
    cut = txt.find(HEAD)
    if cut >= 0:  # replace only this section, keep any sections after it
        nxt = txt.find("\n## ", cut + len(HEAD))
        txt = txt[:cut] + (txt[nxt:] if nxt >= 0 else "")
    res.write_text(txt.rstrip("\n") + "\n\n" + "\n".join(md))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
