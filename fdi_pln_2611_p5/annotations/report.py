from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fdi_pln_2611_p5.annotations.dataset import load_merged_dataset
from fdi_pln_2611_p5.annotations.etiquetados import MergeBundle, merge_etiquetados
from fdi_pln_2611_p5.annotations.report_html import build_report_html
from fdi_pln_2611_p5.annotations.merge import records_to_text_and_labels
from fdi_pln_2611_p5.config import package_path
from fdi_pln_2611_p5.labels import LABEL2ID, PREFIX_TO_ENTITY_TYPE


def _label_distribution(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))


def _merged_json_label_distribution(merged_json: Path) -> dict[str, int]:
    """Cuenta etiquetas por tipo leyendo data/annotations/merged.json."""
    counter: Counter[str] = Counter()
    for sentence in load_merged_dataset(merged_json):
        counter.update(sentence["labels"])
    return dict(counter)


def _confusion_matrix(
    labels_a: list[str], labels_b: list[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for left, right in zip(labels_a, labels_b, strict=True):
        matrix.setdefault(left, Counter())
        matrix[left][right] += 1
    return {row: dict(cols) for row, cols in matrix.items()}


def _count_entities(labels: list[str]) -> dict[str, int]:
    counts = Counter()
    current: str | None = None
    for label in labels:
        if label == "o":
            current = None
            continue
        prefix = label[0] if label else ""
        etype = PREFIX_TO_ENTITY_TYPE.get(prefix, "?")
        if label.endswith("i"):
            counts[etype] += 1
            current = etype
        elif label.endswith("c") and current == etype:
            continue
    return dict(counts)


def _scan_coverage(etiquetados_root: Path) -> list[dict]:
    rows: list[dict] = []
    for json_dir in sorted(etiquetados_root.iterdir()):
        if not json_dir.is_dir():
            continue
        for ann_path in sorted(json_dir.glob("*.json")):
            records = json.loads(ann_path.read_text(encoding="utf-8"))
            _, labels = records_to_text_and_labels(records)
            n = len(labels)
            entity = sum(1 for label in labels if label != "o")
            rows.append(
                {
                    "file": f"{json_dir.name}/{ann_path.name}",
                    "tokens": n,
                    "entity_tokens": entity,
                    "entity_pct": round(100 * entity / max(n, 1), 2),
                }
            )
    return rows


def _collect_analytics(bundle: MergeBundle, etiquetados_root: Path) -> dict:
    merged_labels: list[str] = []
    ann_a_labels: list[str] = []
    ann_b_labels: list[str] = []
    per_frase: list[dict] = []

    for detail in bundle.frase_details:
        merged_labels.extend(detail.merged_labels)
        if len(detail.label_sets) >= 2:
            ann_a_labels.extend(detail.label_sets[0])
            ann_b_labels.extend(detail.label_sets[1])
        per_frase.append(
            {
                "frase_id": detail.frase_id,
                "lote": detail.lote,
                "kappa": detail.kappa,
                "agreement": detail.token_agreement,
                "disagreements": detail.disagreements,
                "text_preview": detail.text[:120].replace("\n", " "),
                "sources": detail.sources,
            }
        )

    confusion = _confusion_matrix(ann_a_labels, ann_b_labels) if ann_a_labels else {}
    all_labels = sorted(set(LABEL2ID.keys()))

    kappas = [f["kappa"] for f in per_frase if f["kappa"] is not None]
    agreements = [f["agreement"] for f in per_frase]

    return {
        "merged_label_dist": _label_distribution(merged_labels),
        "annotator_a_dist": _label_distribution(ann_a_labels),
        "annotator_b_dist": _label_distribution(ann_b_labels),
        "merged_entities": _count_entities(merged_labels),
        "confusion": confusion,
        "confusion_labels": all_labels,
        "per_frase": sorted(per_frase, key=lambda row: row["kappa"] or 0),
        "kappa_values": kappas,
        "agreement_values": agreements,
        "coverage": _scan_coverage(etiquetados_root),
        "report": bundle.report,
    }


def _compact_chart_opts(**extra) -> dict:
    tick = {"font": {"size": 9, "family": "Inter"}, "color": "#94a3b8"}
    grid = {"color": "#f1f5f9", "drawBorder": False}
    base = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {
                "labels": {
                    "boxWidth": 10,
                    "font": {"size": 9, "family": "Inter"},
                    "padding": 10,
                    "color": "#64748b",
                },
            }
        },
        "scales": {
            "x": {"ticks": tick, "grid": grid},
            "y": {"ticks": tick, "grid": grid},
        },
    }
    if "scales" in extra:
        base["scales"] = {**base.get("scales", {}), **extra.pop("scales")}
    base.update(extra)
    return base


def _chart_js_script(
    chart_id: str,
    chart_type: str,
    labels: list,
    datasets: list[dict],
    options: dict | None = None,
) -> str:
    opts = options or {}
    return f"""
    new Chart(document.getElementById('{chart_id}'), {{
      type: '{chart_type}',
      data: {{ labels: {json.dumps(labels)}, datasets: {json.dumps(datasets)} }},
      options: {json.dumps(opts)}
    }});
    """


def generate_annotation_report(
    bundle: MergeBundle | None = None,
    etiquetados_root: Path | None = None,
    output_html: Path | None = None,
    merged_json: Path | None = None,
) -> Path:
    etiquetados_root = etiquetados_root or package_path("data/etiquetados")
    output_html = output_html or package_path(
        "data/annotations/informe_etiquetado.html"
    )
    merged_json = merged_json or package_path("data/annotations/merged.json")

    if bundle is None:
        bundle = merge_etiquetados(
            etiquetados_root=etiquetados_root,
            output_path=merged_json,
        )

    merged_json_label_dist = _merged_json_label_distribution(merged_json)

    data = _collect_analytics(bundle, etiquetados_root)
    data["merged_json_label_dist"] = merged_json_label_dist
    report = data["report"]

    label_names = list(LABEL2ID.keys())
    label_colors = {
        "o": "#cbd5e1",
        "pi": "#6366f1",
        "pc": "#a5b4fc",
        "li": "#10b981",
        "lc": "#6ee7b7",
    }
    dist_colors = [label_colors.get(label, "#64748b") for label in label_names]

    merged_dist = [data["merged_label_dist"].get(label, 0) for label in label_names]
    ann_a_dist = [data["annotator_a_dist"].get(label, 0) for label in label_names]
    ann_b_dist = [data["annotator_b_dist"].get(label, 0) for label in label_names]

    entity_types = sorted(set(data["merged_entities"].keys()) or ["PER", "LOC"])
    entity_counts = [data["merged_entities"].get(t, 0) for t in entity_types]

    kappa_bins = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "≥0.8"]
    kappas = data["kappa_values"]
    kappa_hist = [
        sum(1 for k in kappas if k < 0.2),
        sum(1 for k in kappas if 0.2 <= k < 0.4),
        sum(1 for k in kappas if 0.4 <= k < 0.6),
        sum(1 for k in kappas if 0.6 <= k < 0.8),
        sum(1 for k in kappas if k >= 0.8),
    ]

    charts_js = "\n".join(
        [
            _chart_js_script(
                "chartLabels",
                "bar",
                label_names,
                [
                    {
                        "label": "Fusionado",
                        "data": merged_dist,
                        "backgroundColor": dist_colors,
                        "borderRadius": 4,
                        "borderSkipped": False,
                    },
                    {
                        "label": "Anotador A",
                        "data": ann_a_dist,
                        "backgroundColor": "#e0e7ff",
                        "borderRadius": 4,
                        "borderSkipped": False,
                    },
                    {
                        "label": "Anotador B",
                        "data": ann_b_dist,
                        "backgroundColor": "#f1f5f9",
                        "borderRadius": 4,
                        "borderSkipped": False,
                    },
                ],
                _compact_chart_opts(
                    plugins={
                        "legend": {
                            "position": "bottom",
                            "labels": {"boxWidth": 8, "font": {"size": 8}},
                        }
                    },
                ),
            ),
            _chart_js_script(
                "chartKappaHist",
                "bar",
                kappa_bins,
                [
                    {
                        "label": "Frases",
                        "data": kappa_hist,
                        "backgroundColor": [
                            "#fca5a5",
                            "#fdba74",
                            "#fcd34d",
                            "#a5b4fc",
                            "#6366f1",
                        ],
                        "borderRadius": 6,
                        "borderSkipped": False,
                    }
                ],
                _compact_chart_opts(plugins={"legend": {"display": False}}),
            ),
            _chart_js_script(
                "chartEntities",
                "doughnut",
                entity_types,
                [
                    {
                        "data": entity_counts,
                        "backgroundColor": ["#6366f1", "#10b981", "#f59e0b"],
                        "borderWidth": 0,
                        "hoverOffset": 4,
                    }
                ],
                _compact_chart_opts(
                    cutout="62%",
                    plugins={
                        "legend": {
                            "position": "right",
                            "labels": {"boxWidth": 8, "font": {"size": 8}},
                        }
                    },
                ),
            ),
            _chart_js_script(
                "chartKappaScatter",
                "scatter",
                [],
                [
                    {
                        "label": "κ",
                        "data": [{"x": i, "y": k} for i, k in enumerate(kappas)],
                        "backgroundColor": "rgba(99, 102, 241, 0.55)",
                        "borderColor": "#6366f1",
                        "borderWidth": 1,
                        "pointRadius": 3,
                        "pointHoverRadius": 5,
                    }
                ],
                _compact_chart_opts(
                    plugins={"legend": {"display": False}},
                    scales={
                        "x": {
                            "ticks": {"font": {"size": 7}, "maxTicksLimit": 8},
                            "grid": {"display": False},
                        },
                        "y": {
                            "min": 0,
                            "max": 1,
                            "ticks": {"font": {"size": 8}, "stepSize": 0.2},
                        },
                    },
                ),
            ),
        ]
    )

    page = build_report_html(data, charts_js)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")
    return output_html
