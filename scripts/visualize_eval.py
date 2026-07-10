"""
Visualize eval results as one PDF per example (blog-article style).

Modes:
  --mode alignment    : input structure + GT function text + generated text
  --mode interleaved  : GT narrative (with inline structures) +
                        generated narrative (with inline structures) +
                        Ramachandran score

Usage (geobpe conda env):
  conda activate geobpe
  cd /home/steven/ProteinChamaleon

  python scripts/visualize_eval.py --mode alignment \
      --jsonl /path/to/generation_examples.jsonl \
      --bpe   /home/steven/PT-BPE_ckpts/bpe_post_init.pkl \
      --out-dir /path/to/alignment_pdfs/

  python scripts/visualize_eval.py --mode interleaved \
      --jsonl /path/to/interleaved_generation_examples.jsonl \
      --bpe   /home/steven/PT-BPE_ckpts/bpe_post_init.pkl \
      --out-dir /path/to/interleaved_pdfs/
"""

import argparse, base64, json, os, sys, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
import pickle
import pandas as pd
from io import BytesIO

sys.path.insert(0, "/home/steven/GeoBPE")
from foldingdiff.angles_and_coords import create_new_chain_nerf

STRUCT_COLOR = "#4C72B0"
GT_COLOR     = "#DD8452"


# ── BPE decode ────────────────────────────────────────────────────────────────

def load_bpe(path):
    print(f"Loading BPE from {path} ...")
    with open(path, "rb") as f:
        bpe = pickle.load(f)
    print("BPE loaded.")
    return bpe


def decode_to_repl(bpe, bpe_ids):
    tokenized = bpe.dequantize(bpe_ids)
    return bpe.recover(tokenized)


def repl_to_coords(repl):
    keys = ["phi","psi","omega","tau","N:CA","CA:C","0C:1N","CA:C:1N","C:1N:1CA"]
    defaults = {"N:CA":1.46,"CA:C":1.52,"0C:1N":1.33}
    arrays = {}
    for k in keys:
        try:
            arr = np.asarray(repl.get(k,[]), dtype=float).flatten()
        except Exception:
            arr = np.array([])
        arrays[k] = arr
    lengths = [len(v) for v in arrays.values() if len(v) > 0]
    if not lengths or min(lengths) < 3:
        return None, None
    n = min(lengths)
    df_dict = {}
    for k, arr in arrays.items():
        if len(arr) >= n:
            df_dict[k] = arr[:n].tolist()
        else:
            df_dict[k] = arr.tolist() + [defaults.get(k, 0.0)] * (n - len(arr))
    angles_df = pd.DataFrame(df_dict)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if not create_new_chain_nerf(tmp_path, angles_df):
            return None, angles_df
        ca = []
        with open(tmp_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        coords = np.array(ca) if len(ca) > 3 else None
        return coords, angles_df
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def decode_to_coords(bpe, bpe_ids):
    try:
        repl = decode_to_repl(bpe, bpe_ids)
        coords, _ = repl_to_coords(repl)
        return coords
    except Exception as e:
        print(f"    decode failed: {e}")
        return None


def decode_with_angles(bpe, bpe_ids):
    """Returns (coords, angles_df) — angles_df has phi/psi for Ramachandran."""
    try:
        repl = decode_to_repl(bpe, bpe_ids)
        return repl_to_coords(repl)
    except Exception as e:
        print(f"    decode failed: {e}")
        return None, None


# ── Ramachandran ──────────────────────────────────────────────────────────────

def ramachandran_score(angles_df):
    """Fraction of (phi, psi) pairs in the allowed Ramachandran regions."""
    if angles_df is None or "phi" not in angles_df or "psi" not in angles_df:
        return None
    phi = np.array(angles_df["phi"], dtype=float)
    psi = np.array(angles_df["psi"], dtype=float)
    valid = np.isfinite(phi) & np.isfinite(psi)
    phi, psi = phi[valid], psi[valid]
    # BPE stores dihedrals in [0, 2π]; convert to [-π, π] for standard Ramachandran regions
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    psi = (psi + np.pi) % (2 * np.pi) - np.pi
    if len(phi) == 0:
        return None
    # Favoured regions (radians): alpha-helix and beta-sheet
    # alpha: phi in (-1.57, -0.52), psi in (-1.05,  0.70)
    # beta:  phi in (-2.27, -0.52), psi in ( 0.87,  3.14) or (-3.14, -2.62)
    alpha = ((-1.57 < phi) & (phi < -0.52) & (-1.05 < psi) & (psi < 0.70))
    beta  = ((-2.27 < phi) & (phi < -0.52) & ((psi > 0.87) | (psi < -2.62)))
    allowed = alpha | beta
    return float(allowed.sum()) / len(phi)


def ramachandran_png_b64(angles_dfs, labels, colors):
    """Ramachandran plot for one or more sets of angles."""
    fig, axes = plt.subplots(1, len(angles_dfs), figsize=(4 * len(angles_dfs), 3.5))
    if len(angles_dfs) == 1:
        axes = [axes]
    for ax, adf, label, color in zip(axes, angles_dfs, labels, colors):
        if adf is not None and "phi" in adf and "psi" in adf:
            phi = np.degrees(np.array(adf["phi"], dtype=float))
            psi = np.degrees(np.array(adf["psi"], dtype=float))
            valid = np.isfinite(phi) & np.isfinite(psi)
            ax.scatter(phi[valid], psi[valid], s=6, alpha=0.6, color=color, edgecolors="none")
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.axhline(0, color="#ccc", lw=0.5); ax.axvline(0, color="#ccc", lw=0.5)
        ax.set_xlabel("φ (°)", fontsize=8); ax.set_ylabel("ψ (°)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Structure rendering ────────────────────────────────────────────────────────

def struct_png_b64(coords, title=""):
    fig = plt.figure(figsize=(3.5, 3.2))
    ax = fig.add_subplot(111, projection="3d")
    if coords is not None:
        xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
        ax.plot(xs, ys, zs, color=STRUCT_COLOR, linewidth=1.8, alpha=0.9)
        ax.scatter(xs[0],  ys[0],  zs[0],  color="#2ca02c", s=35, zorder=5, label="N")
        ax.scatter(xs[-1], ys[-1], zs[-1], color="#d62728", s=35, zorder=5, label="C")
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.text2D(0.5, 0.5, "decode failed", ha="center", va="center",
                  transform=ax.transAxes, fontsize=9, color="gray")
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6); ax.set_zlabel("Z", fontsize=6)
    ax.tick_params(labelsize=5); ax.grid(False)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── HTML/CSS ──────────────────────────────────────────────────────────────────

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2.5cm 2.2cm 2.5cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt; color: #888;
    }
}
body {
    font-family: Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1a1a1a;
}
.header {
    background: #f7f7f7;
    border-left: 4px solid #444;
    padding: 10pt 14pt 8pt 14pt;
    margin-bottom: 18pt;
}
.header h1 { margin: 0 0 3pt 0; font-size: 14pt; }
.header .meta { font-family: monospace; font-size: 9pt; color: #555; }
.section-title {
    font-size: 10pt; font-weight: bold;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 2pt 7pt; border-radius: 3pt;
    display: inline-block; margin-bottom: 8pt;
}
.gt-title    { background: #e8f5e9; color: #2e7d32; }
.gen-title   { background: #e3eaf5; color: #1a3a6b; }
.input-title { background: #fce8d5; color: #7a3100; }
.input-box {
    background: #fffaf5;
    border: 1px solid #f0c090;
    border-radius: 4pt;
    padding: 10pt 14pt;
    margin-bottom: 16pt;
    font-family: monospace;
    font-size: 9pt;
    line-height: 1.6;
    color: #4a2800;
    word-break: break-all;
}
.input-box .field { font-weight: bold; color: #7a3100; }
.narrative {
    padding: 10pt 14pt;
    border-radius: 4pt;
    margin-bottom: 20pt;
    text-align: justify;
}
.narrative-gt  { background: #f9fdf9; border: 1px solid #c8e6c9; color: #1b5e20; }
.narrative-gen { background: #f8fafd; border: 1px solid #c5d5ea; color: #0d2b5e; }
.struct-figure {
    text-align: center;
    margin: 14pt auto;
    page-break-inside: avoid;
}
.struct-figure img { width: 52%; border: 1px solid #ddd; border-radius: 4pt; }
.struct-caption { font-size: 8.5pt; color: #666; margin-top: 4pt; font-style: italic; }
.metrics-box {
    background: #fffdf0;
    border: 1px solid #e0d89a;
    border-radius: 4pt;
    padding: 8pt 14pt;
    font-size: 9pt;
    margin-top: 16pt;
}
.metrics-box table { width: 100%; border-collapse: collapse; }
.metrics-box td { padding: 2pt 8pt; }
.metrics-box td:first-child { font-weight: bold; color: #555; width: 40%; }
.rama-figure { text-align: center; margin: 14pt auto; }
.rama-figure img { width: 85%; border: 1px solid #ddd; border-radius: 4pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 18pt 0; }
"""


def he(s):
    return (str(s)
            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            .replace('"',"&quot;"))


def render_segments_html(segments, label, bpe, collect_angles=False):
    """Render a segment list to HTML parts. Returns (html_parts, all_angles_dfs)."""
    parts = []
    all_angles = []
    struct_n = 0
    for seg in segments:
        if seg["type"] == "text":
            parts.append(he(seg["content"]))
        else:
            struct_n += 1
            bpe_ids = seg["bpe_ids"]
            print(f"    [{label}] struct {struct_n}: {len(bpe_ids)} tokens", end="", flush=True)
            if collect_angles:
                coords, angles_df = decode_with_angles(bpe, bpe_ids)
                all_angles.append(angles_df)
            else:
                coords = decode_to_coords(bpe, bpe_ids)
                angles_df = None
            n_ca = len(coords) if coords is not None else 0
            print(f" → {n_ca} Cα")
            b64 = struct_png_b64(coords, title=f"Structure {struct_n}")
            caption = f"Structure {struct_n} &mdash; {len(bpe_ids)} BPE tokens, {n_ca} C&alpha; residues"
            parts.append(
                f"<div class='struct-figure'>"
                f"<img src='data:image/png;base64,{b64}'/>"
                f"<div class='struct-caption'>{caption}</div>"
                f"</div>"
            )
    return parts, all_angles


# ── Per-example HTML builders ─────────────────────────────────────────────────

def build_alignment_html(ex, bpe, metrics=None):
    organism  = he(ex.get("organism",""))
    accession = he(ex.get("accession",""))
    gt_text   = he(ex.get("ground_truth",""))
    gen_text  = he(ex.get("generated",""))
    bpe_ids   = ex.get("struct_bpe_ids", [])

    # Render input structure
    print(f"  Rendering input structure ({len(bpe_ids)} tokens)...")
    coords = decode_to_coords(bpe, bpe_ids) if bpe_ids else None
    n_ca = len(coords) if coords is not None else 0
    struct_b64 = struct_png_b64(coords, title=f"Input structure ({n_ca} Cα)")

    seq_display = he(ex.get("sequence", "")) or f"[not available]"

    parts = [
        f"<div class='header'>"
        f"<h1>{organism}</h1>"
        f"<div class='meta'>Accession: {accession}</div>"
        f"</div>",

        "<div class='section-title input-title'>Input (Prompt)</div>",
        f"<div class='input-box'>"
        f"<span class='field'>Organism:</span> {organism}<br>"
        f"<span class='field'>Sequence:</span> {seq_display}<br>"
        f"<span class='field'>Structure:</span> {len(bpe_ids)} BPE tokens &rarr; {n_ca} C&alpha; residues (rendered below)"
        f"</div>",

        f"<div class='struct-figure'>"
        f"<img src='data:image/png;base64,{struct_b64}'/>"
        f"<div class='struct-caption'>Input structure &mdash; {len(bpe_ids)} BPE tokens, {n_ca} C&alpha; residues</div>"
        f"</div>",

        "<hr>",

        "<div class='section-title gt-title'>Ground Truth</div>",
        f"<div class='narrative narrative-gt'>{gt_text}</div>",

        "<hr>",

        "<div class='section-title gen-title'>Generated</div>",
        f"<div class='narrative narrative-gen'>{gen_text}</div>",
    ]

    if metrics:
        rows = "".join(
            f"<tr><td>{he(k)}</td><td>{v:.4f}</td></tr>"
            for k, v in metrics.items()
        )
        parts.append(
            f"<div class='metrics-box'><table>{rows}</table></div>"
        )

    return wrap_html("".join(parts))


def build_interleaved_html(ex, bpe, metrics=None, scorers=None):
    organism  = he(ex.get("organism",""))
    accession = he(ex.get("accession",""))
    gt_segs   = ex.get("gt_segments", [])
    gen_segs  = ex.get("segments", [])

    # BERTScore on text portions
    gt_text  = " ".join(s["content"] for s in gt_segs  if s["type"] == "text")
    gen_text = " ".join(s["content"] for s in gen_segs if s["type"] == "text")
    if metrics is None:
        metrics = {}
    if gt_text and gen_text and scorers and "bert" in scorers:
        try:
            P, R, F1 = scorers["bert"].score([gen_text], [gt_text])
            metrics["bertscore_f1"]        = float(F1[0])
            metrics["bertscore_precision"] = float(P[0])
            metrics["bertscore_recall"]    = float(R[0])
        except Exception:
            pass

    print(f"  [{accession}] GT segments...")
    gt_parts, gt_angles   = render_segments_html(gt_segs,  "gt",  bpe, collect_angles=True)
    print(f"  [{accession}] Gen segments...")
    gen_parts, gen_angles = render_segments_html(gen_segs, "gen", bpe, collect_angles=True)

    # Ramachandran — compare GT vs generated
    rama_html = ""

    def collect_phi_psi(angles_list):
        phi_all, psi_all = [], []
        scores = []
        for adf in angles_list:
            if adf is None:
                continue
            s = ramachandran_score(adf)
            if s is not None:
                scores.append(s)
            if "phi" in adf:
                phi_all.extend(np.array(adf["phi"], dtype=float).tolist())
                psi_all.extend(np.array(adf["psi"], dtype=float).tolist())
        return phi_all, psi_all, scores

    gt_phi,  gt_psi,  gt_scores  = collect_phi_psi([a for a in gt_angles  if a is not None])
    gen_phi, gen_psi, gen_scores = collect_phi_psi([a for a in gen_angles if a is not None])

    avg_gt_rama  = float(np.mean(gt_scores))  if gt_scores  else None
    avg_gen_rama = float(np.mean(gen_scores)) if gen_scores else None

    if metrics is not None:
        if avg_gt_rama  is not None: metrics["ramachandran_allowed_gt"]  = avg_gt_rama
        if avg_gen_rama is not None: metrics["ramachandran_allowed_gen"] = avg_gen_rama

    if gt_phi or gen_phi:
        angle_sets, labels, colors = [], [], []
        if gt_phi:
            angle_sets.append(pd.DataFrame({"phi": gt_phi,  "psi": gt_psi}))
            labels.append("Ground truth")
            colors.append(GT_COLOR)
        if gen_phi:
            angle_sets.append(pd.DataFrame({"phi": gen_phi, "psi": gen_psi}))
            labels.append("Generated")
            colors.append(STRUCT_COLOR)
        rama_b64 = ramachandran_png_b64(angle_sets, labels, colors)
        caption_parts = []
        if avg_gt_rama  is not None: caption_parts.append(f"GT: {avg_gt_rama*100:.1f}%")
        if avg_gen_rama is not None: caption_parts.append(f"Gen: {avg_gen_rama*100:.1f}%")
        caption = "Ramachandran — % in allowed regions: " + ", ".join(caption_parts) if caption_parts else "Ramachandran plot"
        rama_html = (
            f"<div class='rama-figure'>"
            f"<img src='data:image/png;base64,{rama_b64}'/>"
            f"<div class='struct-caption'>{caption}</div>"
            f"</div>"
        )

    seq_display = he(ex.get("sequence", ""))

    parts = [
        f"<div class='header'>"
        f"<h1>{organism}</h1>"
        f"<div class='meta'>Accession: {accession}</div>"
        f"</div>",

        "<div class='section-title input-title'>Input (Prompt)</div>",
        f"<div class='input-box'>"
        f"<span class='field'>Organism:</span> {organism}<br>"
        f"<span class='field'>Sequence:</span> {seq_display}"
        f"</div>",

        "<hr>",

        "<div class='section-title gt-title'>Ground Truth</div>",
        f"<div class='narrative narrative-gt'>{''.join(gt_parts)}</div>",

        "<hr>",

        "<div class='section-title gen-title'>Generated</div>",
        f"<div class='narrative narrative-gen'>{''.join(gen_parts)}</div>",

        rama_html,
    ]

    if metrics:
        rows = "".join(
            f"<tr><td>{he(k)}</td><td>{v:.4f}</td></tr>"
            for k, v in metrics.items()
        )
        parts.append(
            f"<div class='metrics-box'><table>{rows}</table></div>"
        )

    return wrap_html("".join(parts))


def wrap_html(body):
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


# ── Main ──────────────────────────────────────────────────────────────────────

def load_scorers(mode):
    scorers = {}
    if mode == "alignment":
        from bert_score import BERTScorer
        from rouge_score import rouge_scorer as _rs
        print("Loading BERTScorer (RoBERTa-large)...")
        scorers["bert"] = BERTScorer(lang="en")
        scorers["rouge"] = _rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    elif mode == "interleaved":
        from bert_score import BERTScorer
        print("Loading BERTScorer (RoBERTa-large)...")
        scorers["bert"] = BERTScorer(lang="en")
    return scorers


def compute_alignment_metrics(ex, scorers):
    metrics = {}
    gt  = ex.get("ground_truth", "")
    gen = ex.get("generated", "")
    if not gt or not gen:
        return metrics
    try:
        P, R, F1 = scorers["bert"].score([gen], [gt])
        metrics["bertscore_f1"]        = float(F1[0])
        metrics["bertscore_precision"] = float(P[0])
        metrics["bertscore_recall"]    = float(R[0])
    except Exception:
        pass
    try:
        s = scorers["rouge"].score(gt, gen)
        metrics["rouge1"] = s["rouge1"].fmeasure
        metrics["rouge2"] = s["rouge2"].fmeasure
        metrics["rougeL"] = s["rougeL"].fmeasure
    except Exception:
        pass
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      required=True, choices=["alignment","interleaved"])
    parser.add_argument("--examples-dir", required=True,
                        help="Directory containing per-example subdirs (e.g. eval_results/alignment/)")
    parser.add_argument("--bpe",       required=True)
    parser.add_argument("--n",         type=int, default=None)
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    example_dirs = sorted([d for d in examples_dir.iterdir() if d.is_dir()])
    if args.n:
        example_dirs = example_dirs[:args.n]
    print(f"Found {len(example_dirs)} examples (mode={args.mode})")

    bpe     = load_bpe(args.bpe)
    scorers = load_scorers(args.mode)

    from weasyprint import HTML

    for i, ex_dir in enumerate(example_dirs):
        acc      = ex_dir.name
        json_path = ex_dir / f"{acc}.json"
        pdf_path  = ex_dir / f"{acc}.pdf"

        if not json_path.exists():
            print(f"[{i+1}] {acc}: no json, skipping")
            continue

        # Resume: skip if PDF already exists
        if pdf_path.exists():
            print(f"[{i+1}] {acc}: already done, skipping")
            continue

        print(f"\n[{i+1}/{len(example_dirs)}] {acc}")
        ex = json.loads(json_path.read_text())

        if args.mode == "alignment":
            metrics = compute_alignment_metrics(ex, scorers)
            html    = build_alignment_html(ex, bpe, metrics=metrics)
        else:
            metrics = {}
            html    = build_interleaved_html(ex, bpe, metrics=metrics, scorers=scorers)

        HTML(string=html).write_pdf(str(pdf_path))

        # Save metrics back into the json
        ex["metrics"] = metrics
        json_path.write_text(json.dumps(ex, indent=2))

        print(f"  → {pdf_path.name} + metrics saved")

    print(f"\nDone. PDFs written to {examples_dir}/")


if __name__ == "__main__":
    main()
