from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def generate_grid_search_html(results_path: Path, output_path: Path) -> Path:
    """Genera un informe HTML con los resultados del grid search."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results: list[dict] = payload["results"]
    best: dict = payload["best"]

    sorted_results = sorted(results, key=lambda r: r["test_loss"])
    min_loss = sorted_results[0]["test_loss"]
    max_loss = sorted_results[-1]["test_loss"]

    lr_avg = _avg_by_key(results, "learning_rate", "test_loss")
    bs_avg = _avg_by_key(results, "batch_size", "test_loss")
    best_lr = min(lr_avg, key=lr_avg.get)
    best_bs = min(bs_avg, key=bs_avg.get)

    html = _build_html(sorted_results, best, min_loss, max_loss, lr_avg, bs_avg, best_lr, best_bs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _avg_by_key(results: list[dict], group_key: str, value_key: str) -> dict:
    groups: dict = {}
    for r in results:
        k = r[group_key]
        groups.setdefault(k, []).append(r[value_key])
    return {k: sum(v) / len(v) for k, v in groups.items()}


def _loss_color(loss: float, min_loss: float, max_loss: float) -> str:
    if max_loss == min_loss:
        ratio = 0.0
    else:
        ratio = (loss - min_loss) / (max_loss - min_loss)
    r = int(50 + 180 * ratio)
    g = int(200 - 150 * ratio)
    return f"rgb({r},{g},80)"


def _bar(value: float, min_val: float, max_val: float, width: int = 180) -> str:
    ratio = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    bar_width = max(4, int(width * ratio))
    color = _loss_color(value, min_val, max_val)
    return (
        f'<div style="background:{color};width:{bar_width}px;height:14px;'
        f'border-radius:3px;display:inline-block;"></div>'
    )


def _build_html(
    sorted_results: list[dict],
    best: dict,
    min_loss: float,
    max_loss: float,
    lr_avg: dict,
    bs_avg: dict,
    best_lr: float,
    best_bs: int,
) -> str:
    rows = ""
    for i, r in enumerate(sorted_results):
        is_best = r["learning_rate"] == best["learning_rate"] and r["batch_size"] == best["batch_size"]
        bg = "#e8f5e9" if is_best else ("#f9f9f9" if i % 2 == 0 else "#ffffff")
        marker = " ★" if is_best else ""
        bar = _bar(r["test_loss"], min_loss, max_loss)
        rows += (
            f'<tr style="background:{bg}">'
            f'<td style="font-weight:{"bold" if is_best else "normal"}">{r["learning_rate"]}{marker}</td>'
            f"<td>{r['batch_size']}</td>"
            f"<td>{r['train_loss']:.4f}</td>"
            f"<td>{r['test_loss']:.4f}</td>"
            f"<td>{bar}</td>"
            f"</tr>\n"
        )

    lr_rows = "".join(
        f"<tr><td>{'<b>' + str(k) + '</b>' if k == best_lr else k}</td><td>{v:.4f}</td></tr>"
        for k, v in sorted(lr_avg.items(), key=lambda x: x[1])
    )
    bs_rows = "".join(
        f"<tr><td>{'<b>' + str(k) + '</b>' if k == best_bs else k}</td><td>{v:.4f}</td></tr>"
        for k, v in sorted(bs_avg.items(), key=lambda x: x[1])
    )

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Grid Search — Exploración de Hiperparámetros</title>
<style>
  body {{ font-family: sans-serif; max-width: 860px; margin: 40px auto; color: #222; }}
  h1 {{ color: #1a237e; }} h2 {{ color: #283593; border-bottom: 2px solid #c5cae9; padding-bottom:4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th {{ background: #3949ab; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; }}
  .summary {{ background: #e8f5e9; border-left: 4px solid #43a047; padding: 12px 16px;
              border-radius:4px; margin-bottom:24px; }}
  .analysis {{ display: flex; gap: 32px; }}
  .analysis table {{ flex: 1; }}
  .note {{ background:#fff8e1; border-left:4px solid #fbc02d; padding:10px 14px; border-radius:4px; }}
  footer {{ color:#888; font-size:0.85em; margin-top:40px; }}
</style>
</head>
<body>
<h1>Exploración de Hiperparámetros — Grid Search</h1>
<p>Generado el {date}</p>

<div class="summary">
  <strong>Mejor configuración:</strong>
  learning_rate = <code>{best["learning_rate"]}</code> &nbsp;|&nbsp;
  batch_size = <code>{best["batch_size"]}</code> &nbsp;|&nbsp;
  test_loss = <code>{best["test_loss"]:.4f}</code>
</div>

<h2>Resultados de las {len(sorted_results)} combinaciones</h2>
<table>
  <tr>
    <th>Learning rate</th><th>Batch size</th>
    <th>Train loss</th><th>Test loss</th><th>Comparativa</th>
  </tr>
  {rows}
</table>

<h2>Análisis por hiperparámetro</h2>
<div class="analysis">
  <div>
    <h3>Por learning rate (test_loss medio)</h3>
    <table><tr><th>LR</th><th>Media test_loss</th></tr>{lr_rows}</table>
  </div>
  <div>
    <h3>Por batch size (test_loss medio)</h3>
    <table><tr><th>Batch</th><th>Media test_loss</th></tr>{bs_rows}</table>
  </div>
</div>

<div class="note">
  <strong>Observación:</strong> El mejor learning rate en promedio fue <code>{best_lr}</code>
  y el mejor batch size fue <code>{best_bs}</code>.
  Los resultados del grid search están guardados en <code>data/grid_search_results.json</code>
  para consulta posterior.
</div>

<footer>Práctica 5 — LM Causal + NER | fdi-pln-2611</footer>
</body>
</html>"""
