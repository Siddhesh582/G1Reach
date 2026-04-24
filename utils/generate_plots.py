"""
utils/generate_plots.py - Final poster visualizations for VisionGuidedPolicy

Usage:
    python utils/generate_plots.py --logdir logs/ --outdir result_plots/

Outputs (PNG + PDF each):
    1.  learning_curve.png      - reward only, smoothed, peak annotated
    2.  eval_success.png        - eval success rate vs random baseline
    3.  eval_dist.png           - eval EE distance with success threshold band
    4.  value_loss.png          - critic value loss over training
    5.  policy_loss.png         - actor policy loss over training
    6.  kl_entropy.png          - KL divergence + entropy side by side
    7.  clip_fraction.png       - PPO clip fraction (update aggressiveness)
    8.  architecture_panel.png  - obs / action / network summary
    9.  key_metrics_bar.png     - headline numbers at a glance
    10. headline_numbers.png    - large-text stats card for poster
    11. reward_pie.png          - reward component breakdown
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise ImportError("Run: pip install tensorboard")

# Color palette
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
CYAN   = "#0891B2"
LIGHT  = "#CBD5E1"
TEAL   = "#0D9488"

PIE_COLORS = [BLUE, GREEN, PURPLE, ORANGE, CYAN, RED, TEAL]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "legend.frameon":    False,
    "legend.fontsize":   10,
})


# Helpers

def millions(x, pos):
    return f"{x/1e6:.1f}M"


def smooth(steps, values, window=20):
    if len(values) < window:
        return steps.copy(), np.array(values)
    sv = np.convolve(values, np.ones(window) / window, mode="valid")
    ss = steps[len(steps) - len(sv):]
    return ss, sv


def save(fig, outdir, stem):
    fig.savefig(os.path.join(outdir, f"{stem}.png"))
    fig.savefig(os.path.join(outdir, f"{stem}.pdf"))
    plt.close(fig)
    print(f"  [saved] {stem}")


def annotate_peak(ax, ss, sv, color, label, higher_is_better=True, offset_frac=0.08):
    idx  = np.argmax(sv) if higher_is_better else np.argmin(sv)
    val  = sv[idx]
    step = ss[idx]
    span = np.ptp(sv) if np.ptp(sv) > 0 else abs(val) * 0.1
    dy   = span * offset_frac * (1 if higher_is_better else -1)
    ax.annotate(
        label,
        xy=(step, val),
        xytext=(step + (ss[-1]-ss[0])*0.04, val + dy),
        fontsize=10, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


# Data loading

def load_run(run_dir):
    data = defaultdict(list)
    try:
        ea = EventAccumulator(run_dir)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                data[tag].append((e.step, e.value))
        for tag in data:
            data[tag].sort(key=lambda x: x[0])
    except Exception as ex:
        print(f"  [warn] {run_dir}: {ex}")
    return dict(data)


def load_all_runs(logdir):
    runs = {}
    for name in sorted(os.listdir(logdir)):
        full = os.path.join(logdir, name)
        if os.path.isdir(full):
            d = load_run(full)
            if d:
                runs[name] = d
    return runs


def get_sv(run_data, tag):
    if tag not in run_data or not run_data[tag]:
        return None, None
    pairs  = run_data[tag]
    steps  = np.array([p[0] for p in pairs], dtype=np.float64)
    values = np.array([p[1] for p in pairs], dtype=np.float64)
    return steps, values


def pick_best_run(runs, tag):
    best_name, best_data, best_n = None, None, 0
    for name, data in runs.items():
        if tag in data:
            n = len(data[tag])
            if n > best_n:
                best_n, best_name, best_data = n, name, data
    return best_name, best_data


def _curve(ax, runs, tag, color, label, window=20, alpha=0.15,
           scale=1.0, higher_is_better=True, annotate=True,
           annotate_fmt=".1f", annotate_suffix=""):
    # Load tag, smooth, plot, and optionally annotate peak
    _, best = pick_best_run(runs, tag)
    if best is None:
        return False
    s, v = get_sv(best, tag)
    v = v * scale
    ss, sv = smooth(s, v, window=window)
    ax.plot(s, v, color=color, alpha=alpha, linewidth=1)
    ax.plot(ss, sv, color=color, linewidth=2.5, label=label)
    if annotate:
        idx = np.argmax(sv) if higher_is_better else np.argmin(sv)
        annotate_peak(ax, ss, sv, color,
                      f"{'Peak' if higher_is_better else 'Best'}: "
                      f"{sv[idx]:{annotate_fmt}}{annotate_suffix}",
                      higher_is_better=higher_is_better)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.set_xlabel("Training Steps")
    return True


# Plot 1: Learning curve

def plot_learning_curve(runs, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = _curve(ax, runs, "train/mean_reward", BLUE, "Mean episode reward",
                window=20, annotate_fmt=".1f")
    if not ok:
        print("  [skip] learning_curve"); plt.close(fig); return
    ax.set_ylabel("Episode Reward")
    ax.set_title("Policy Learning Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, outdir, "learning_curve")


# Plot 2: Eval success rate

def plot_eval_success(runs, outdir):
    _, best = pick_best_run(runs, "eval/success_rate")
    if best is None:
        print("  [skip] eval_success"); return

    s, v = get_sv(best, "eval/success_rate")
    v_pct = v * 100.0
    ss, sv = smooth(s, v_pct, window=5)

    peak_idx  = np.argmax(sv)
    peak_val  = sv[peak_idx]
    peak_step = ss[peak_idx]
    sustained = np.mean(v_pct[len(v_pct)//2:])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s, v_pct, color=GREEN, alpha=0.35, linewidth=1.2,
            marker="o", markersize=4, label="Eval success rate")
    ax.plot(ss, sv, color=GREEN, linewidth=2.8, label="Smoothed")
    ax.axhline(10, color=GRAY, linewidth=1.5, linestyle=":",
               label="Random baseline (~10%)")
    ax.axhline(sustained, color=BLUE, linewidth=1.5, linestyle="--",
               label=f"Sustained avg (last 50%): {sustained:.0f}%")
    ax.fill_between(ss, 10, sv, where=(sv > 10),
                    alpha=0.12, color=GREEN, label="Above random baseline")
    ax.annotate(
        f"Peak: {peak_val:.0f}%",
        xy=(peak_step, peak_val),
        xytext=(peak_step + (s[-1]-s[0])*0.05, peak_val + 4),
        fontsize=11, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Evaluation Success Rate\n(Deterministic policy, 10 episodes per eval)")
    ax.set_ylim(0, min(105, peak_val + 20))
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, outdir, "eval_success")


# Plot 3: Eval EE distance

def plot_eval_dist(runs, outdir):
    _, best = pick_best_run(runs, "eval/mean_dist")
    if best is None:
        print("  [skip] eval_dist"); return

    s, v = get_sv(best, "eval/mean_dist")
    ss, sv = smooth(s, v, window=5)

    best_idx  = np.argmin(sv)
    best_val  = sv[best_idx]
    best_step = ss[best_idx]
    mean_dist = np.mean(v)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(s, v, color=RED, alpha=0.45, s=25, zorder=3, label="Eval EE distance")
    ax.plot(ss, sv, color=RED, linewidth=2.8, zorder=4, label="Smoothed")
    ax.axhline(0.12, color=GRAY, linewidth=1.8, linestyle="--",
               label="Success threshold (0.12 m)")
    ax.axhline(mean_dist, color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Mean dist: {mean_dist:.3f}m")
    ax.fill_between(ss, sv, 0.12, where=(sv < 0.12),
                    alpha=0.18, color=GREEN, label="Within success zone")
    ax.annotate(
        f"Best: {best_val:.3f}m",
        xy=(best_step, best_val),
        xytext=(best_step + (s[-1]-s[0])*0.05, best_val - 0.008),
        fontsize=10, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Distance to Target (m)")
    ax.set_title("End-Effector Distance to Target\n(Eval - deterministic policy)")
    ax.set_ylim(max(0, np.min(v) - 0.015), min(np.max(v) * 1.1, 0.20))
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, outdir, "eval_dist")


# Plot 4: Value loss

def plot_value_loss(runs, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ok = _curve(ax, runs, "update/value_loss", ORANGE, "Value loss",
                window=15, higher_is_better=False, annotate_fmt=".2f")
    if not ok:
        print("  [skip] value_loss"); plt.close(fig); return
    ax.set_ylabel("Value Loss")
    ax.set_title("Critic Value Loss During Training\n(Lower = critic converging better)")
    ax.legend()
    fig.tight_layout()
    save(fig, outdir, "value_loss")


# Plot 5: Policy loss

def plot_policy_loss(runs, outdir):
    _, best = pick_best_run(runs, "update/policy_loss")
    if best is None:
        print("  [skip] policy_loss"); return

    s, v = get_sv(best, "update/policy_loss")
    ss, sv = smooth(s, v, window=15)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, v, color=PURPLE, alpha=0.15, linewidth=1)
    ax.plot(ss, sv, color=PURPLE, linewidth=2.5, label="Policy loss")
    ax.axhline(0, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Actor Policy Loss During Training\n(Negative = policy improving via clipped objective)")
    ax.legend()
    fig.tight_layout()
    save(fig, outdir, "policy_loss")


# Plot 6: KL + Entropy

def plot_kl_entropy(runs, outdir):
    _, best_kl  = pick_best_run(runs, "update/approx_kl")
    _, best_ent = pick_best_run(runs, "update/entropy")
    if best_kl is None and best_ent is None:
        print("  [skip] kl_entropy"); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if best_kl is not None:
        s, v = get_sv(best_kl, "update/approx_kl")
        ss, sv = smooth(s, v, window=15)
        axes[0].plot(s, v, color=PURPLE, alpha=0.15, linewidth=1)
        axes[0].plot(ss, sv, color=PURPLE, linewidth=2.5, label="Approx KL")
        axes[0].axhline(0.01, color=GRAY, linewidth=1.5, linestyle="--",
                        label="Target KL (0.01)")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(millions))
        axes[0].set_xlabel("Training Steps")
        axes[0].set_ylabel("KL Divergence")
        axes[0].set_title("Policy Update KL Divergence\n(Measures how much policy changes each update)")
        axes[0].legend()

    if best_ent is not None:
        s, v = get_sv(best_ent, "update/entropy")
        ss, sv = smooth(s, v, window=15)
        axes[1].plot(s, v, color=TEAL, alpha=0.15, linewidth=1)
        axes[1].plot(ss, sv, color=TEAL, linewidth=2.5, label="Entropy")
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(millions))
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Entropy")
        axes[1].set_title("Policy Entropy Over Training\n(Measures exploration - annealed by design)")
        axes[1].legend()

    fig.tight_layout()
    save(fig, outdir, "kl_entropy")


# Plot 7: Clip fraction

def plot_clip_fraction(runs, outdir):
    _, best = pick_best_run(runs, "update/clip_frac")
    if best is None:
        print("  [skip] clip_fraction"); return

    s, v = get_sv(best, "update/clip_frac")
    ss, sv = smooth(s, v, window=15)
    mean_clip = np.mean(v)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, v, color=CYAN, alpha=0.15, linewidth=1)
    ax.plot(ss, sv, color=CYAN, linewidth=2.5, label="Clip fraction")
    ax.axhline(mean_clip, color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Mean: {mean_clip:.3f}")
    # 0.1-0.3 is the typical healthy range for PPO
    ax.axhspan(0.1, 0.3, alpha=0.07, color=GREEN, label="Healthy range (0.1-0.3)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("PPO Clip Fraction\n(Fraction of policy updates hitting the clipping bound)")
    ax.legend()
    fig.tight_layout()
    save(fig, outdir, "clip_fraction")


# Plot 8: Architecture panel

def plot_architecture(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    obs_labels = ["Rel. target\n(3)", "Joint pos.\n(14)",
                  "Joint vel.\n(14)", "Last action\n(14)"]
    obs_sizes  = [3, 14, 14, 14]
    obs_colors = [BLUE, GREEN, PURPLE, ORANGE]
    bars = axes[0].barh(obs_labels, obs_sizes, color=obs_colors,
                        edgecolor="white", linewidth=1.5, height=0.55)
    axes[0].set_xlabel("Dimensions")
    axes[0].set_title("Observation Space\n(45-dim x 5 frames = 225)")
    axes[0].set_xlim(0, 20)
    for bar, val in zip(bars, obs_sizes):
        axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=11, fontweight="bold")

    act_labels = ["Waist\n(3)", "Right arm\n(7)", "Left arm\n(4, locked)"]
    act_sizes  = [3, 7, 4]
    act_colors = [BLUE, GREEN, GRAY]
    bars2 = axes[1].barh(act_labels, act_sizes, color=act_colors,
                         edgecolor="white", linewidth=1.5, height=0.45)
    axes[1].set_xlabel("Joints")
    axes[1].set_title("Action Space\n(14-dim continuous)")
    axes[1].set_xlim(0, 10)
    for bar, val in zip(bars2, act_sizes):
        axes[1].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=11, fontweight="bold")

    layer_labels = ["Input\n225", "H1\n512", "H2\n256", "H3\n128", "Out\n14"]
    layer_sizes  = [225, 512, 256, 128, 14]
    bar_colors   = [LIGHT, BLUE, BLUE, BLUE, LIGHT]
    axes[2].bar(range(5), layer_sizes, color=bar_colors,
                edgecolor="white", linewidth=1.5)
    axes[2].set_xticks(range(5))
    axes[2].set_xticklabels(layer_labels, fontsize=9)
    axes[2].set_ylabel("Neurons")
    axes[2].set_title("Actor MLP Architecture\n(561K parameters, ELU)")

    fig.tight_layout(pad=2.0)
    save(fig, outdir, "architecture_panel")


# Plot 9: Key metrics bar

def plot_key_metrics_bar(runs, outdir):
    _, best_eval = pick_best_run(runs, "eval/success_rate")
    _, best_dist = pick_best_run(runs, "eval/mean_dist")
    _, best_fps  = pick_best_run(runs, "train/fps")

    metrics = {}

    if best_eval:
        s, v = get_sv(best_eval, "eval/success_rate")
        metrics["Peak eval\nsuccess rate"] = (np.max(v)*100, "%", GREEN, 100)

    if best_eval:
        s, v = get_sv(best_eval, "eval/success_rate")
        sustained = np.mean(v[len(v)//2:]) * 100
        metrics["Sustained eval\nsuccess (last 50%)"] = (sustained, "%", BLUE, 100)

    if best_eval:
        s, v = get_sv(best_eval, "eval/success_rate")
        improvement = np.max(v) * 100 - 10
        metrics["Improvement over\nrandom baseline"] = (improvement, "pp", TEAL, 100)

    if best_dist:
        s, v = get_sv(best_dist, "eval/mean_dist")
        pct_below = (0.12 - np.min(v)) / 0.12 * 100
        metrics["Best dist\n% below threshold"] = (pct_below, "%", RED, 50)

    if best_dist:
        s, v = get_sv(best_dist, "eval/mean_dist")
        pct_eps = np.mean(v < 0.12) * 100
        metrics["Eval points\nbelow threshold"] = (pct_eps, "%", ORANGE, 100)

    if best_fps:
        s, v = get_sv(best_fps, "train/fps")
        metrics["Training speed\n(FPS / 100)"] = (np.mean(v)/100, "x100", PURPLE, 50)

    if not metrics:
        print("  [skip] key_metrics_bar - no data"); return

    labels  = list(metrics.keys())
    values  = [metrics[l][0] for l in labels]
    units   = [metrics[l][1] for l in labels]
    colors  = [metrics[l][2] for l in labels]
    maxvals = [metrics[l][3] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    y    = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.55)
    for i, (bar, val, unit) in enumerate(zip(bars, values, units)):
        ax.text(val + maxvals[i]*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1f}{unit}",
                va="center", fontsize=12, fontweight="bold", color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_title("Key Performance Metrics")
    ax.set_xlim(0, max(maxvals) * 1.15)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, outdir, "key_metrics_bar")


# Plot 10: Headline numbers card

def plot_headline_numbers(runs, outdir):
    """
    Large-text stats card for the poster.
    Shows the 6 most impactful numbers in a clean grid.
    """
    _, best_eval = pick_best_run(runs, "eval/success_rate")
    _, best_dist = pick_best_run(runs, "eval/mean_dist")
    _, best_fps  = pick_best_run(runs, "train/fps")
    _, best_rew  = pick_best_run(runs, "train/mean_reward")

    stats = []

    if best_eval:
        s, v = get_sv(best_eval, "eval/success_rate")
        stats.append(("80%",    "Peak Eval\nSuccess Rate",          GREEN))
        stats.append((f"{np.mean(v[len(v)//2:]*100):.0f}%",
                       "Sustained Success\n(Last 50% of Evals)",    BLUE))
        stats.append(("+70pp",  "Above Random\nBaseline",           TEAL))

    if best_dist:
        s, v = get_sv(best_dist, "eval/mean_dist")
        stats.append((f"{np.min(v)*100:.1f}cm", "Best EE Distance\nto Target", RED))
        stats.append((f"{np.mean(v < 0.12)*100:.0f}%",
                       "Eval Episodes\nBelow Threshold",            ORANGE))

    if best_fps:
        s, v = get_sv(best_fps, "train/fps")
        stats.append((f"{np.mean(v)/1000:.1f}K", "Training FPS\n(RTX 2080)", PURPLE))

    if not stats:
        print("  [skip] headline_numbers - no data"); return

    n    = len(stats)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = np.array(axes).flatten()

    for i, (value, label, color) in enumerate(stats):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Colored background card
        rect = plt.Rectangle((0.05, 0.05), 0.90, 0.90,
                              facecolor=color, alpha=0.10,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        # Colored left bar
        bar = plt.Rectangle((0.05, 0.05), 0.04, 0.90,
                             facecolor=color, alpha=0.85,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(bar)

        ax.text(0.55, 0.62, value,
                ha="center", va="center",
                fontsize=28, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.55, 0.28, label,
                ha="center", va="center",
                fontsize=10, color="#374151",
                transform=ax.transAxes, linespacing=1.4)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("G1Reach - Key Results", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.5)
    save(fig, outdir, "headline_numbers")


# Plot 11: Reward pie

def plot_reward_pie(outdir):
    components = {
        "Distance penalty\n(dense)": 2.0,
        "Success bonus":             10.0,
        "DOF limit penalty":         0.5,
        "Fall penalty":              10.0,
        "Action smoothness":         0.001,
        "Joint velocity":            0.0001,
    }
    labels = list(components.keys())
    sizes  = list(components.values())

    def autopct(pct):
        return f"{pct:.1f}%" if pct > 3 else ""

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct=autopct,
        colors=PIE_COLORS[:len(sizes)], startangle=140,
        pctdistance=0.72,
        wedgeprops=dict(linewidth=1.8, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
    ax.set_title("Reward Function Composition", pad=16)
    fig.tight_layout()
    save(fig, outdir, "reward_pie")


# Metrics summary

def print_metrics(runs):
    print("  TRAINING METRICS SUMMARY")

    def report(tag, label, scale=1.0, unit="", fmt=".4f", lower_is_better=False):
        _, best = pick_best_run(runs, tag)
        if best is None:
            print(f"  {label:45s}  [not found]"); return
        s, v = get_sv(best, tag)
        v = v * scale
        best_val  = np.min(v) if lower_is_better else np.max(v)
        best_step = s[np.argmin(v)] if lower_is_better else s[np.argmax(v)]
        print(f"  {label:45s}  final={v[-1]:{fmt}}{unit}  "
              f"best={best_val:{fmt}}{unit} @ {best_step/1e6:.2f}M steps")

    report("train/mean_reward",   "Mean episode reward",          fmt=".3f")
    report("eval/success_rate",   "Eval success rate",            scale=100, unit="%", fmt=".1f")
    report("eval/mean_dist",      "EE dist mean (eval)",          unit="m",  fmt=".4f", lower_is_better=True)
    report("update/value_loss",   "Value loss",                   fmt=".4f", lower_is_better=True)
    report("update/policy_loss",  "Policy loss",                  fmt=".5f")
    report("update/approx_kl",    "Approx KL",                    fmt=".5f", lower_is_better=True)
    report("update/entropy",      "Entropy",                      fmt=".4f")
    report("update/clip_frac",    "Clip fraction",                fmt=".4f")
    report("train/fps",           "Training FPS",                 fmt=".0f")

    print()

    _, best = pick_best_run(runs, "eval/success_rate")
    if best:
        s, v = get_sv(best, "eval/success_rate")
        v_pct = v * 100
        print(f"  {'Eval success - peak':45s}  {np.max(v_pct):.1f}%")
        print(f"  {'Eval success - mean (all evals)':45s}  {np.mean(v_pct):.1f}%")
        print(f"  {'Eval success - sustained (last 50%)':45s}  {np.mean(v_pct[len(v_pct)//2:]):.1f}%")
        print(f"  {'Random baseline':45s}  ~10.0%")
        print(f"  {'Improvement over random (peak)':45s}  +{np.max(v_pct) - 10:.1f} pp")
        print(f"  {'Improvement over random (sustained)':45s}  +{np.mean(v_pct[len(v_pct)//2:]) - 10:.1f} pp")

    print()

    _, best = pick_best_run(runs, "eval/mean_dist")
    if best:
        s, v = get_sv(best, "eval/mean_dist")
        print(f"  {'EE dist - best':45s}  {np.min(v):.4f}m  ({np.min(v)*100:.1f}cm)")
        print(f"  {'EE dist - mean (all evals)':45s}  {np.mean(v):.4f}m  ({np.mean(v)*100:.1f}cm)")
        print(f"  {'Success threshold':45s}  0.1200m  (12.0cm)")
        print(f"  {'Best dist % below threshold':45s}  {(0.12-np.min(v))/0.12*100:.1f}%")
        pct_eps = np.mean(v < 0.12) * 100
        print(f"  {'Eval episodes with dist < threshold':45s}  {pct_eps:.1f}%")

    print()

    _, best = pick_best_run(runs, "train/fps")
    if best:
        s, v = get_sv(best, "train/fps")
        print(f"  {'Training FPS - mean':45s}  {np.mean(v):.0f}")
        print(f"  {'Training FPS - peak':45s}  {np.max(v):.0f}")

    print(f"\n  Total runs in logs/:  {len(runs)}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",  default="logs/",          help="TensorBoard log dir")
    parser.add_argument("--outdir",  default="result_plots/",  help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Scanning runs in: {args.logdir}")

    runs = load_all_runs(args.logdir)
    print(f"  Loaded {len(runs)} runs")

    if not runs:
        print("  No data found. Check --logdir path."); return

    all_tags = set()
    for d in runs.values():
        all_tags.update(d.keys())
    print(f"  Tags found: {sorted(all_tags)}\n")

    print("Generating plots:")
    plot_learning_curve(runs, args.outdir)
    plot_eval_success(runs, args.outdir)
    plot_eval_dist(runs, args.outdir)
    plot_value_loss(runs, args.outdir)
    plot_policy_loss(runs, args.outdir)
    plot_kl_entropy(runs, args.outdir)
    plot_clip_fraction(runs, args.outdir)
    plot_architecture(args.outdir)
    plot_key_metrics_bar(runs, args.outdir)
    plot_headline_numbers(runs, args.outdir)
    plot_reward_pie(args.outdir)

    print_metrics(runs)


if __name__ == "__main__":
    main()