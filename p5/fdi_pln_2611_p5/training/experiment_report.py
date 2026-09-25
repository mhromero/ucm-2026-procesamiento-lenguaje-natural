"""HTML report generation for the causal LM experiment exploration suite."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


def generate_experiment_html(payload: dict, output_path: Path) -> Path:
    """Write an HTML experiment report from an in-memory results payload.

    Args:
        payload: Experiment results dict (same schema as the JSON output).
        output_path: Destination HTML file path.

    Returns:
        The ``output_path`` written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_html(payload), encoding="utf-8")
    return output_path


def generate_experiment_html_from_json(results_path: Path, output_path: Path) -> Path:
    """Load experiment JSON from disk and generate the HTML report.

    Args:
        results_path: Path to ``experiment_results.json``.
        output_path: Destination HTML file path.

    Returns:
        The ``output_path`` written.
    """
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return generate_experiment_html(payload, output_path)


def _svg_learning_curve(
    history: list[dict], width: int = 420, height: int = 200
) -> str:
    """Render an SVG train/val loss curve for one experiment."""
    if not history:
        return "<p>Sin historial.</p>"
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    all_vals = train + val
    y_min = min(all_vals) * 0.95
    y_max = max(all_vals) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1.0

    pad_l, pad_r, pad_t, pad_b = 44, 16, 12, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_pos(i: int) -> float:
        if len(epochs) == 1:
            return pad_l + plot_w / 2
        return pad_l + (i / (len(epochs) - 1)) * plot_w

    def y_pos(v: float) -> float:
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    def polyline(vals: list[float], color: str) -> str:
        pts = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>'

    grid = ""
    for i in range(5):
        gy = pad_t + (i / 4) * plot_h
        gv = y_max - (i / 4) * (y_max - y_min)
        grid += (
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{gv:.2f}</text>'
        )

    xlabels = "".join(
        f'<text x="{x_pos(i):.1f}" y="{height - 8}" text-anchor="middle" '
        f'font-size="10" fill="#666">{e}</text>'
        for i, e in enumerate(epochs)
    )

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" role="img">
  {grid}
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#999"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#999"/>
  {polyline(train, "#1976d2")}
  {polyline(val, "#e65100")}
  {xlabels}
  <text x="{pad_l + plot_w / 2:.0f}" y="{height - 2}" text-anchor="middle" font-size="11" fill="#333">Época</text>
  <text x="12" y="{pad_t + plot_h / 2:.0f}" transform="rotate(-90 12 {pad_t + plot_h / 2:.0f})" text-anchor="middle" font-size="11" fill="#333">Loss</text>
  <rect x="{pad_l + plot_w - 120}" y="{pad_t + 4}" width="110" height="36" fill="white" fill-opacity="0.85" stroke="#ccc"/>
  <line x1="{pad_l + plot_w - 108}" y1="{pad_t + 16}" x2="{pad_l + plot_w - 88}" y2="{pad_t + 16}" stroke="#1976d2" stroke-width="2.5"/>
  <text x="{pad_l + plot_w - 82}" y="{pad_t + 20}" font-size="10">train</text>
  <line x1="{pad_l + plot_w - 108}" y1="{pad_t + 30}" x2="{pad_l + plot_w - 88}" y2="{pad_t + 30}" stroke="#e65100" stroke-width="2.5"/>
  <text x="{pad_l + plot_w - 82}" y="{pad_t + 34}" font-size="10">val (Alice)</text>
</svg>"""


def _comparison_bars(results: list[dict], width: int = 520, height: int = 220) -> str:
    """Render an SVG bar chart comparing best val_loss across experiments."""
    sorted_r = sorted(results, key=lambda r: r["best_epoch"]["val_loss"])
    losses = [r["best_epoch"]["val_loss"] for r in sorted_r]
    y_min, y_max = min(losses) * 0.9, max(losses) * 1.1
    if y_max <= y_min:
        y_max = y_min + 0.1
    pad_l, pad_b = 48, 56
    plot_w = width - pad_l - 20
    plot_h = height - pad_b - 20
    bar_w = plot_w / max(len(sorted_r), 1) * 0.55
    bars = ""
    for i, row in enumerate(sorted_r):
        cx = pad_l + (i + 0.5) * (plot_w / len(sorted_r))
        h = (row["best_epoch"]["val_loss"] - y_min) / (y_max - y_min) * plot_h
        y = pad_b + plot_h - h
        color = "#43a047" if i == 0 else "#5c6bc0"
        bars += (
            f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" fill="{color}" rx="3"/>'
            f'<text x="{cx:.1f}" y="{height - 28}" text-anchor="middle" '
            f'font-size="9" fill="#333">{html.escape(row["id"])}</text>'
            f'<text x="{cx:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#111">{row["best_epoch"]["val_loss"]:.3f}</text>'
        )
    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{pad_l}" y="14" font-size="12" fill="#333">Mejor val_loss por experimento</text>
  <line x1="{pad_l}" y1="{pad_b + plot_h}" x2="{pad_l + plot_w}" y2="{pad_b + plot_h}" stroke="#999"/>
  {bars}
</svg>"""


def _history_table(history: list[dict]) -> str:
    """Render a compact per-epoch loss table for one experiment."""
    rows = ""
    best_val = min(h["val_loss"] for h in history)
    for h in history:
        star = " ★" if h["val_loss"] == best_val else ""
        rows += (
            f"<tr><td>{h['epoch']}</td>"
            f"<td>{h['train_loss']:.4f}</td>"
            f"<td>{h['val_loss']:.4f}{star}</td></tr>"
        )
    return f"""<table class="hist">
      <thead><tr><th>Época</th><th>Train loss</th><th>Val loss</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _build_html(payload: dict) -> str:
    """Assemble the full experiment exploration HTML document."""
    experiments: list[dict] = payload["experiments"]
    best: dict = payload["best"]
    methodology = payload.get("methodology", {})
    fixed = methodology.get("fixed_hyperparams", {})
    date = payload.get("generated_at", "")[:19].replace("T", " ")

    summary_rows = ""
    for exp in sorted(experiments, key=lambda e: e["best_epoch"]["val_loss"]):
        is_best = exp["id"] == best["id"]
        cfg = exp["config"]
        reused = (
            f"<br><span class='meta'>↳ mismos resultados que {html.escape(exp['reused_from'])}</span>"
            if exp.get("reused_from")
            else ""
        )
        summary_rows += f"""<tr class="{"best-row" if is_best else ""}">
          <td><strong>{html.escape(exp["id"])}</strong>{" ★" if is_best else ""}{reused}</td>
          <td>{html.escape(exp["name"])}</td>
          <td>{html.escape(cfg.get("corpus_train", "—"))}</td>
          <td>{cfg["window_size"]}</td>
          <td>{exp["dataset_stats"]["train_windows"]:,}</td>
          <td>{exp["best_epoch"]["val_loss"]:.4f}</td>
          <td>{exp["best_epoch"]["epoch"]}</td>
        </tr>"""

    exp_sections = ""
    for exp in experiments:
        cfg = exp["config"]
        exp_sections += f"""
<section class="experiment">
  <h2>{html.escape(exp["name"])} <code>{html.escape(exp["id"])}</code></h2>
  <p class="meta"><strong>Variable:</strong> {html.escape(exp["variable"])}</p>
  {"<p class='meta'><strong>Nota:</strong> Resultados reutilizados de <code>" + html.escape(exp["reused_from"]) + "</code> (config idéntica).</p>" if exp.get("reused_from") else ""}
  <p class="question"><strong>Pregunta:</strong> {html.escape(exp["research_question"])}</p>
  <p class="hint"><strong>Hipótesis / lectura posible:</strong> {html.escape(exp["expected_insight"])}</p>

  <div class="grid-2">
    <div>
      <h3>Configuración</h3>
      <table class="cfg">
        <tr><td>corpus train (BPE)</td><td>{html.escape(cfg.get("corpus_train", "—"))}</td></tr>
        <tr><td>BPE vocab (real)</td><td>{exp.get("bpe_vocab_actual", cfg["vocab_size"])}</td></tr>
        <tr><td>caché / pesos</td><td><code>{html.escape(exp.get("tokenizer_cache_dir", "—"))}</code><br>
            <span class="meta">model.pth · experiment_config.json</span></td></tr>
        <tr><td>extra_max_books</td><td>{cfg.get("extra_max_books", "—")}</td></tr>
        <tr><td>window_size</td><td>{cfg["window_size"]}</td></tr>
        <tr><td>vocab_size</td><td>{cfg["vocab_size"]}</td></tr>
        <tr><td>n_blocks</td><td>{cfg["n_blocks"]}</td></tr>
        <tr><td>n_heads</td><td>{cfg["n_heads"]}</td></tr>
        <tr><td>d_model</td><td>{cfg["d_model"]}</td></tr>
        <tr><td>dropout</td><td>{cfg["dropout"]}</td></tr>
        <tr><td>lr / batch</td><td>{cfg["learning_rate"]} / {cfg["batch_size"]}</td></tr>
      </table>
      <h3>Corpus → ventanas</h3>
      <table class="cfg">
        <tr><td>Tokens train</td><td>{exp["dataset_stats"]["train_tokens"]:,}</td></tr>
        <tr><td>Tokens test (Alice)</td><td>{exp["dataset_stats"]["test_tokens"]:,}</td></tr>
        <tr><td>Ventanas train</td><td>{exp["dataset_stats"]["train_windows"]:,}</td></tr>
        <tr><td>Batches / época</td><td>{exp["dataset_stats"]["steps_per_epoch"]:,}</td></tr>
      </table>
    </div>
    <div class="chart-box">
      <h3>Curvas de aprendizaje</h3>
      {_svg_learning_curve(exp["history"])}
    </div>
  </div>
  <h3>Histórico por época</h3>
  {_history_table(exp["history"])}
  <p class="final">Loss final: train={exp["train_loss"]:.4f}, val={exp["val_loss"]:.4f}
     · Mejor val={exp["best_epoch"]["val_loss"]:.4f} (época {exp["best_epoch"]["epoch"]})</p>
</section>
"""

    fixed_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in fixed.items()
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Exploración de experimentos — LM causal</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 20px; color: #222; line-height: 1.5; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #c5cae9; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 2em; }}
  h3 {{ color: #3949ab; font-size: 1em; margin-top: 1.2em; }}
  .summary-box {{ background: #e8eaf6; border-left: 4px solid #3949ab; padding: 14px 18px; border-radius: 4px; margin: 20px 0; }}
  .method {{ background: #f5f5f5; padding: 14px 18px; border-radius: 4px; font-size: 0.95em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92em; }}
  th {{ background: #3949ab; color: #fff; padding: 8px 10px; text-align: left; }}
  td {{ border: 1px solid #e0e0e0; padding: 6px 10px; }}
  tr.best-row {{ background: #e8f5e9; }}
  .experiment {{ border: 1px solid #c5cae9; border-radius: 8px; padding: 16px 20px; margin: 28px 0; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .question {{ background: #fff8e1; padding: 10px 12px; border-radius: 4px; }}
  .hint {{ color: #555; font-size: 0.9em; }}
  .chart-box {{ text-align: center; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .final {{ font-weight: 600; margin-top: 12px; }}
  table.hist {{ max-width: 360px; }}
  table.cfg td:first-child {{ font-weight: 500; width: 45%; }}
  footer {{ color: #888; font-size: 0.85em; margin-top: 48px; }}
</style>
</head>
<body>
<h1>Exploración de hiperparámetros y arquitectura</h1>
<p class="meta">Generado: {html.escape(date)} UTC · 8 experimentos · {methodology.get("epochs_per_run", "?")} épocas por entrenamiento único</p>

<div class="summary-box">
  <strong>Mejor experimento:</strong> <code>{html.escape(best["id"])}</code> — {html.escape(best["name"])}<br>
  <strong>Mejor val_loss:</strong> {best["best_epoch"]["val_loss"]:.4f} (época {best["best_epoch"]["epoch"]})<br>
  <strong>Ventanas train:</strong> {best["dataset_stats"]["train_windows"]:,}
</div>

<h2>Metodología</h2>
<div class="method">
  <p>{html.escape(methodology.get("description", ""))}</p>
  <p><strong>Criterio de comparación:</strong> {html.escape(methodology.get("metric_selection", ""))}</p>
  <table>
    <tr><th colspan="2">Hiperparámetros fijos (todos los experimentos)</th></tr>
    {fixed_rows}
  </table>
</div>

<h2>Comparativa global</h2>
<div style="text-align:center">{_comparison_bars(experiments)}</div>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Nombre</th><th>Corpus train</th><th>Ventana</th>
      <th>Ventanas train</th><th>Mejor val_loss</th><th>Época</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>

<h2>Detalle por experimento</h2>
{exp_sections}

<footer>Práctica 5 — fdi-pln-2611 · JSON: <code>data/experiment_results.json</code></footer>
</body>
</html>"""
