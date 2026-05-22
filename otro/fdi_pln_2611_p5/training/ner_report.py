from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def generate_ner_report(
    history: list[dict],
    confusion: dict,
    model_cfg: dict,
    ner_cfg: dict,
    best_epoch: int,
    output_path: Path,
) -> Path:
    """Genera informe HTML del entrenamiento NER con curvas y matriz de confusión."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html(history, confusion, model_cfg, ner_cfg, best_epoch)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _loss_color(loss: float, min_loss: float, max_loss: float) -> str:
    if max_loss == min_loss:
        return "rgb(80,160,80)"
    t = (loss - min_loss) / (max_loss - min_loss)
    r = int(80 + 160 * t)
    g = int(160 - 100 * t)
    return f"rgb({r},{g},80)"


def _bar(value: float, max_value: float, color: str, width: int = 120) -> str:
    px = max(2, int(width * value / max(max_value, 1e-9)))
    return f'<span style="display:inline-block;width:{px}px;height:12px;background:{color};border-radius:2px;vertical-align:middle;"></span>'


def _confusion_cell(count: int, row_total: int) -> str:
    ratio = count / max(row_total, 1)
    intensity = int(40 + 200 * ratio)
    if count == 0:
        bg = "#f8f8f8"
        color = "#bbb"
    else:
        bg = f"rgb({255 - intensity // 2},{255 - intensity},{255 - intensity // 2})"
        color = "#111" if ratio < 0.6 else "#fff"
    return f'<td style="background:{bg};color:{color};text-align:center;padding:6px 10px;font-weight:{"bold" if ratio>0.5 else "normal"}">{count}</td>'


def _build_html(
    history: list[dict],
    confusion: dict,
    model_cfg: dict,
    ner_cfg: dict,
    best_epoch: int,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])
    per_class = confusion.get("per_class", {})

    losses = [r["val_loss"] for r in history]
    min_loss = min(losses) if losses else 0.0
    max_loss = max(losses) if losses else 1.0
    best_row = next((r for r in history if r["epoch"] == best_epoch), history[-1] if history else {})
    best_metric = ner_cfg.get("best_metric", "val_loss")
    has_macro_f1 = any("macro_f1_non_o" in r for r in history)
    has_span_f1 = any("span_f1" in r for r in history)

    # --- training curves table ---
    curve_rows = ""
    for r in history:
        star = " ★" if r["epoch"] == best_epoch else ""
        color = _loss_color(r["val_loss"], min_loss, max_loss)
        bar = _bar(r["val_loss"], max_loss, color)
        acc_bar = _bar(r["overall_acc"], 1.0, "rgb(80,120,200)")
        rec_bar = _bar(r["entity_recall"], 1.0, "rgb(180,80,80)")
        macro_cell = ""
        if has_macro_f1:
            mf1 = r.get("macro_f1_non_o", 0.0)
            macro_cell = (
                f'<td>{_bar(mf1, 1.0, "rgb(120,80,180)")} {_pct(mf1)}</td>'
            )
        span_cell = ""
        if has_span_f1:
            sf1 = r.get("span_f1", 0.0)
            span_cell = f'<td>{_bar(sf1, 1.0, "rgb(80,140,200)")} {_pct(sf1)}</td>'
        row_style = 'style="background:#fffde7"' if r["epoch"] == best_epoch else ""
        curve_rows += f"""<tr {row_style}>
          <td style="text-align:center">{r["epoch"]}{star}</td>
          <td style="text-align:right">{_fmt(r["train_loss"])}</td>
          <td>{bar} {_fmt(r["val_loss"])}</td>
          <td>{acc_bar} {_pct(r["overall_acc"])}</td>
          <td>{rec_bar} {_pct(r["entity_recall"])}</td>
          {macro_cell}
          {span_cell}
          <td style="text-align:center">{r["n_pred_entities"]}</td>
          <td style="text-align:center">{r["n_gold_entities"]}</td>
        </tr>"""

    # --- confusion matrix ---
    header_cells = "".join(
        f'<th style="padding:6px 10px;background:#e8f5e9">pred: {lbl}</th>' for lbl in labels
    )
    conf_rows = ""
    for i, lbl in enumerate(labels):
        row_total = sum(matrix[i]) if matrix else 0
        cells = "".join(_confusion_cell(matrix[i][j], row_total) for j in range(len(labels)))
        conf_rows += f'<tr><th style="padding:6px 10px;background:#e8f5e9;text-align:right">true: {lbl}</th>{cells}</tr>'

    # --- per-class metrics table ---
    class_rows = ""
    for lbl in labels:
        m = per_class.get(lbl, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f = m.get("f1", 0.0)
        s = m.get("support", 0)
        f1_color = f"rgb({int(220-160*f)},{int(80+140*f)},80)"
        class_rows += f"""<tr>
          <td style="font-weight:bold;padding:5px 10px">{lbl}</td>
          <td style="text-align:right;padding:5px 10px">{_pct(p)}</td>
          <td style="text-align:right;padding:5px 10px">{_pct(r)}</td>
          <td style="text-align:right;padding:5px 10px;background:{f1_color};color:#fff;font-weight:bold">{_pct(f)}</td>
          <td style="text-align:right;padding:5px 10px">{s}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe NER — PLN 2611</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #4caf50; padding-bottom: 8px; }}
  h2 {{ color: #388e3c; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #e8f5e9; padding: 6px 10px; text-align: left; border: 1px solid #c8e6c9; }}
  td {{ border: 1px solid #e0e0e0; padding: 4px 8px; }}
  .summary {{ background: #f1f8e9; border-left: 4px solid #66bb6a; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
  .conclusions {{ background: #fff8e1; border-left: 4px solid #ffd54f; padding: 16px; border-radius: 4px; margin: 20px 0; min-height: 80px; }}
  .meta {{ color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Informe de Entrenamiento NER</h1>
<p class="meta">Generado automáticamente · {now}</p>

<div class="summary">
  <strong>Mejor epoch:</strong> {best_epoch} (criterio: {best_metric}) &nbsp;|&nbsp;
  <strong>Val loss:</strong> {_fmt(best_row.get("val_loss", 0))} &nbsp;|&nbsp;
  <strong>Accuracy:</strong> {_pct(best_row.get("overall_acc", 0))} &nbsp;|&nbsp;
  <strong>Entity token accuracy:</strong> {_pct(best_row.get("entity_token_acc", best_row.get("entity_recall", 0)))} &nbsp;|&nbsp;
  <strong>Accuracy global:</strong> {_pct(best_row.get("overall_acc", 0))}
  {f' &nbsp;|&nbsp; <strong>Macro F1 (sin o):</strong> {_pct(best_row.get("macro_f1_non_o", 0))}' if has_macro_f1 else ''}
  {f' &nbsp;|&nbsp; <strong>Span F1:</strong> {_pct(best_row.get("span_f1", 0))}' if has_span_f1 else ''}
  {f' &nbsp;|&nbsp; <strong>Span F1 (informe):</strong> {_pct(confusion.get("span_f1", 0))}' if confusion.get("span_f1") is not None else ''}
  {f' &nbsp;|&nbsp; <strong>Restricción acc ≥</strong> {_pct(ner_cfg.get("min_overall_acc", 0))}' if ner_cfg.get("min_overall_acc") else ''}
</div>

<h2>Configuración del modelo</h2>
<table>
  <tr><th>Parámetro</th><th>Valor</th></tr>
  <tr><td>d_model</td><td>{model_cfg.get("d_model")}</td></tr>
  <tr><td>n_blocks</td><td>{model_cfg.get("n_blocks")}</td></tr>
  <tr><td>n_heads</td><td>{model_cfg.get("n_heads")}</td></tr>
  <tr><td>window_size</td><td>{model_cfg.get("window_size")}</td></tr>
  <tr><td>dropout</td><td>{model_cfg.get("dropout")}</td></tr>
  <tr><td>epochs NER</td><td>{ner_cfg.get("epochs")}</td></tr>
  <tr><td>batch_size NER</td><td>{ner_cfg.get("batch_size")}</td></tr>
  <tr><td>learning_rate NER</td><td>{ner_cfg.get("learning_rate")}</td></tr>
  <tr><td>val_ratio</td><td>{ner_cfg.get("val_ratio")}</td></tr>
  <tr><td>loss</td><td>{ner_cfg.get("loss", "ce")}</td></tr>
  <tr><td>best_metric</td><td>{ner_cfg.get("best_metric", "val_loss")}</td></tr>
</table>

<h2>Curvas de entrenamiento</h2>
<table>
  <thead>
    <tr>
      <th>Epoch</th><th>Train loss</th><th>Val loss</th>
      <th>Accuracy</th><th>Entity Recall</th>
      {"<th>Macro F1</th>" if has_macro_f1 else ""}
      {"<th>Span F1</th>" if has_span_f1 else ""}
      <th>Pred. entidades</th><th>Gold entidades</th>
    </tr>
  </thead>
  <tbody>{curve_rows}</tbody>
</table>

<h2>Matriz de confusión (conjunto de validación)</h2>
<p class="meta">Filas = etiqueta real · Columnas = etiqueta predicha · Intensidad = proporción respecto al total de la fila</p>
<table>
  <thead><tr><th></th>{header_cells}</tr></thead>
  <tbody>{conf_rows}</tbody>
</table>

<h2>Métricas por clase</h2>
<table>
  <thead>
    <tr><th>Etiqueta</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
  </thead>
  <tbody>{class_rows}</tbody>
</table>

<h2>Análisis y conclusiones</h2>
<div class="conclusions">
  <!-- Completar manualmente después del entrenamiento -->
  <p><em>(Espacio para conclusiones manuales: qué etiquetas se detectan mejor, errores más frecuentes en la matriz de confusión, comportamiento del modelo en entidades de persona vs. lugar, etc.)</em></p>
</div>

</body>
</html>"""
