from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from fdi_pln_2611_p5.annotations.merge import (
    cohen_kappa,
    extract_frase_records,
    load_assignments,
    merge_sentence_labels,
    normalize_merge_label,
    records_to_word_labels,
)
from fdi_pln_2611_p5.annotations.dataset import save_merged_dataset
from fdi_pln_2611_p5.config import package_path


@dataclass
class FraseMergeResult:
    frase_id: int
    text: str
    lote: str
    json_indices: list[int]
    sources: list[str]
    label_sets: list[list[str]]
    merged_labels: list[str]
    token_agreement: float
    kappa: float | None
    disagreements: int


@dataclass
class MergeBundle:
    report: dict
    sentences: list[dict]
    frase_details: list[FraseMergeResult] = field(default_factory=list)


def _json_dir_candidates(json_idx: int) -> list[str]:
    n = json_idx + 1
    return [
        f"json_{n:02d}",
        f"json_{n}",
        f"json{n}",
        f"json{n:02d}",
    ]


def resolve_etiquetados_json_dir(etiquetados_root: Path, json_idx: int) -> Path | None:
    for name in _json_dir_candidates(json_idx):
        path = etiquetados_root / name
        if path.is_dir():
            return path
    return None


def resolve_annotation_file(json_dir: Path, parte_suffix: str) -> Path | None:
    """parte_suffix: 'p1' (parte1) o 'p2' (parte2)."""
    matches = sorted(json_dir.glob(f"*_{parte_suffix}.json"))
    if matches:
        return matches[0]
    return None


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_lote(
    etiquetados_root: Path,
    assignments_path: Path,
    lote: str,
    parte_suffix: str,
    frase_id_offset: int = 0,
) -> tuple[list[dict], list[FraseMergeResult], dict]:
    granularidad, frases, assignments = load_assignments(assignments_path)
    frase_texts = [frase.lower() for frase in frases]

    merged_sentences: list[dict] = []
    frase_details: list[FraseMergeResult] = []
    kappas: list[float] = []
    token_agreements: list[float] = []
    skipped_no_pair = 0
    skipped_unlabeled = 0

    json_indices_by_frase: dict[int, list[int]] = {i: [] for i in range(len(frases))}
    for json_idx, frase_indices in enumerate(assignments):
        for frase_idx in frase_indices:
            json_indices_by_frase[frase_idx].append(json_idx)

    for frase_idx, json_indices in json_indices_by_frase.items():
        if len(json_indices) != 2:
            logger.warning(
                "Frase {} en {} tiene {} JSON (se esperaban 2).",
                frase_idx,
                lote,
                len(json_indices),
            )
            skipped_no_pair += 1
            continue

        word_sets: list[tuple[list[str], list[str]]] = []
        sources: list[str] = []

        for json_idx in json_indices:
            json_dir = resolve_etiquetados_json_dir(etiquetados_root, json_idx)
            if json_dir is None:
                logger.warning("No existe carpeta para json index {}", json_idx)
                continue
            ann_path = resolve_annotation_file(json_dir, parte_suffix)
            if ann_path is None:
                logger.warning("Sin anotación {} en {}", parte_suffix, json_dir)
                continue

            records = load_records(ann_path)
            chunk = extract_frase_records(records, frase_texts[frase_idx])
            if chunk is None:
                logger.warning("Frase {} no hallada en {}", frase_idx, ann_path.name)
                continue
            _, tokens, labels = records_to_word_labels(chunk)
            word_sets.append((tokens, labels))
            sources.append(f"{json_dir.name}/{ann_path.name}")

        if len(word_sets) < 2:
            skipped_unlabeled += 1
            continue

        label_sets = [labels for _, labels in word_sets]
        tokens = word_sets[0][0]
        if word_sets[0][0] != word_sets[1][0]:
            logger.warning(
                "Frase {} en {}: tokens distintos entre anotadores.",
                frase_idx,
                lote,
            )
        if len(label_sets[0]) != len(label_sets[1]):
            logger.warning(
                "Frase {} longitudes distintas: {} vs {} palabras",
                frase_idx,
                len(label_sets[0]),
                len(label_sets[1]),
            )
            continue

        norm_a = [normalize_merge_label(l) for l in label_sets[0]]
        norm_b = [normalize_merge_label(l) for l in label_sets[1]]
        kappa = cohen_kappa(norm_a, norm_b)
        merged_labels, agreement = merge_sentence_labels(label_sets)
        disagreements = sum(1 for a, b in zip(norm_a, norm_b) if a != b)

        kappas.append(kappa)
        token_agreements.append(agreement)
        global_id = frase_id_offset + frase_idx

        frase_details.append(
            FraseMergeResult(
                frase_id=global_id,
                text=frase_texts[frase_idx],
                lote=lote,
                json_indices=json_indices,
                sources=sources,
                label_sets=label_sets,
                merged_labels=merged_labels,
                token_agreement=agreement,
                kappa=kappa,
                disagreements=disagreements,
            )
        )
        merged_sentences.append(
            {
                "frase_id": global_id,
                "lote": lote,
                "text": frase_texts[frase_idx],
                "tokens": tokens,
                "labels": merged_labels,
                "sources": sources,
                "kappa": kappa,
                "token_agreement": agreement,
            }
        )

    lote_report = {
        "lote": lote,
        "granularidad": granularidad,
        "n_frases_corpus": len(frases),
        "n_frases_fusionadas": len(merged_sentences),
        "skipped_no_pair": skipped_no_pair,
        "skipped_unlabeled": skipped_unlabeled,
        "mean_token_agreement": sum(token_agreements) / max(len(token_agreements), 1),
        "mean_cohen_kappa": sum(kappas) / max(len(kappas), 1),
        "pairwise_kappas": kappas,
    }
    return merged_sentences, frase_details, lote_report


def merge_etiquetados(
    etiquetados_root: Path | None = None,
    parte1_assignments: Path | None = None,
    parte2_assignments: Path | None = None,
    lote_9frases_assignments: Path | None = None,
    output_path: Path | None = None,
) -> MergeBundle:
    etiquetados_root = etiquetados_root or package_path("data/etiquetados")
    parte1_assignments = parte1_assignments or package_path(
        "data/asignaciones/alice_jsons_parte1/asignaciones.json"
    )
    parte2_assignments = parte2_assignments or package_path(
        "data/asignaciones/alice_jsons_parte2/asignaciones.json"
    )
    output_path = output_path or package_path("data/annotations/merged.json")

    s1, d1, r1 = merge_lote(
        etiquetados_root, parte1_assignments, "parte1", "p1", frase_id_offset=0
    )
    offset = r1["n_frases_corpus"]
    s2, d2, r2 = merge_lote(
        etiquetados_root,
        parte2_assignments,
        "parte2",
        "p2",
        frase_id_offset=offset,
    )

    lotes = [r1, r2]
    all_sentences = s1 + s2
    all_details = d1 + d2

    if lote_9frases_assignments is not None:
        offset = r1["n_frases_corpus"] + r2["n_frases_corpus"]
        s3, d3, r3 = merge_lote(
            etiquetados_root,
            lote_9frases_assignments,
            "lote_9frases",
            "p2",
            frase_id_offset=offset,
        )
        all_sentences += s3
        all_details += d3
        lotes.append(r3)
        logger.info(
            "Lote 9 frases (json_14/json_15): {} frases fusionadas",
            r3["n_frases_fusionadas"],
        )

    all_kappas = [k for r in lotes for k in r["pairwise_kappas"]]
    all_agreements = [f.token_agreement for f in all_details]

    report = {
        "source": str(etiquetados_root),
        "n_frases": len(all_sentences),
        "mean_token_agreement": sum(all_agreements) / max(len(all_agreements), 1),
        "mean_cohen_kappa": sum(all_kappas) / max(len(all_kappas), 1),
        "pairwise_kappas": all_kappas,
        "lotes": lotes,
    }

    save_merged_dataset(output_path, all_sentences)
    logger.info("Fusión etiquetados → {}", output_path)
    logger.info(
        "Frases fusionadas: {} (parte1={}, parte2={})",
        len(all_sentences),
        r1["n_frases_fusionadas"],
        r2["n_frases_fusionadas"],
    )
    logger.info("κ medio global: {:.3f}", report["mean_cohen_kappa"])

    return MergeBundle(
        report=report, sentences=all_sentences, frase_details=all_details
    )
