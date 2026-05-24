"""Merge dual-annotator JSON labels into a consensus NER dataset.

Normalizes labels, resolves disagreements with BIO-aware rules, and computes
inter-annotator agreement metrics before writing merged sentence records.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from fdi_pln_2611_p5.annotations.ner_dataset import save_merged_dataset
from fdi_pln_2611_p5.paths import require_dir, require_file


@dataclass
class FraseMergeResult:
    """Per-sentence merge outcome with annotator comparison details."""

    frase_id: int
    text: str
    json_indices: list[int]
    sources: list[str]
    label_sets: list[list[str]]
    merged_labels: list[str]
    token_agreement: float
    kappa: float | None
    disagreements: int


@dataclass
class MergeBundle:
    """Full merge output: summary report, sentences, and per-sentence details."""

    report: dict
    sentences: list[dict]
    frase_details: list[FraseMergeResult] = field(default_factory=list)


VALID_LABELS = frozenset({"o", "pi", "pc", "li", "lc"})
LABEL_TYPOS = {"ps": "pi", "o ": "o", " o": "o"}


def normalize_merge_label(label: str | None) -> str:
    """Normalize a raw annotator label to a valid merge vocabulary tag.

    Valid labels are ``o``, ``pi``, ``pc``, ``li``, and ``lc``; anything else
    maps to ``o``.

    Args:
        label: Raw label string from an annotation record.

    Returns:
        Normalized label in ``VALID_LABELS``.
    """
    if not label:
        return "o"
    cleaned = label.strip().lower()
    cleaned = LABEL_TYPOS.get(cleaned, cleaned)
    if cleaned not in VALID_LABELS:
        return "o"
    return cleaned


def _entity_prefix(label: str) -> str | None:
    """Return the entity-type prefix (``p`` or ``l``) for a non-``o`` label."""
    if label == "o":
        return None
    return label[0]


def coerce_bio_label(label: str, prev_label: str) -> str:
    """Coerce a label to BIO-style ``i`` (initial) or ``c`` (continuation).

    Args:
        label: Candidate normalized label.
        prev_label: Previous token label in the sentence.

    Returns:
        BIO-adjusted label consistent with ``prev_label``.
    """
    label = normalize_merge_label(label)
    if label == "o":
        return "o"
    prefix = _entity_prefix(label)
    assert prefix is not None
    prev_label = normalize_merge_label(prev_label)
    if prev_label == "o" or _entity_prefix(prev_label) != prefix:
        return f"{prefix}i"
    return f"{prefix}c"


def merge_disagreeing_labels(votes: list[str], prev_label: str) -> str:
    """Resolve conflicting annotator votes for a single token.

    Args:
        votes: Raw label votes from annotators.
        prev_label: Previous merged token label.

    Returns:
        Consensus label after normalization and BIO coercion.
    """
    votes = [normalize_merge_label(v) for v in votes]
    if len(set(votes)) == 1:
        return coerce_bio_label(votes[0], prev_label)

    non_o = [v for v in votes if v != "o"]
    if not non_o:
        return "o"
    if len(non_o) == 1:
        return coerce_bio_label(non_o[0], prev_label)

    prefixes = {_entity_prefix(v) for v in non_o}
    if len(prefixes) == 1:
        return coerce_bio_label(non_o[0], prev_label)

    return coerce_bio_label(Counter(non_o).most_common(1)[0][0], prev_label)


def records_to_text_and_labels(records: list[dict]) -> tuple[str, list[str]]:
    """Expand annotation records to character-level text and labels.

    Legacy helper used for reports and sentence boundary matching.

    Args:
        records: Annotation records with ``clave`` and ``valor`` fields.

    Returns:
        Tuple of reconstructed text and one label per character.
    """
    text = "".join(item["clave"] for item in records)
    labels: list[str] = []
    for item in records:
        label = normalize_merge_label(item.get("valor"))
        labels.extend([label] * len(item["clave"]))
    return text, labels


def records_to_word_labels(records: list[dict]) -> tuple[str, list[str], list[str]]:
    """Convert records to one label per annotation unit (word, space, or punctuation).

    Args:
        records: Annotation records with ``clave`` and ``valor`` fields.

    Returns:
        Tuple of full text, token units, and per-unit labels.
    """
    tokens = [item["clave"] for item in records]
    labels = [normalize_merge_label(item.get("valor")) for item in records]
    text = "".join(tokens)
    return text, tokens, labels


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two label sequences of equal length.

    Args:
        labels_a: Labels from annotator A.
        labels_b: Labels from annotator B.

    Returns:
        Kappa coefficient in ``[0, 1]`` (1.0 for empty input).

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Las secuencias deben tener la misma longitud.")
    if not labels_a:
        return 1.0

    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    matrix: dict[tuple[str, str], int] = Counter()
    for left, right in zip(labels_a, labels_b, strict=True):
        matrix[(left, right)] += 1

    observed = sum(matrix[(cat, cat)] for cat in categories) / n
    marg_a = {
        c: sum(matrix[(c, other)] for other in categories) / n for c in categories
    }
    marg_b = {
        c: sum(matrix[(other, c)] for other in categories) / n for c in categories
    }
    expected = sum(marg_a[c] * marg_b[c] for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def normalize_annotation_text(text: str) -> str:
    """Normalize curly quotes and apostrophes for fuzzy sentence matching.

    Args:
        text: Raw annotation text.

    Returns:
        Text with Unicode quote variants replaced by ASCII equivalents.
    """
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def merge_sentence_labels(label_sets: list[list[str]]) -> tuple[list[str], float]:
    """Merge per-annotator label sequences for one sentence.

    Args:
        label_sets: One label list per annotator, all of equal length.

    Returns:
        Tuple of merged labels and mean raw token agreement (for pairs only).

    Raises:
        ValueError: If annotator sequences differ in length.
    """
    if not label_sets:
        return [], 1.0
    length = len(label_sets[0])
    if any(len(labels) != length for labels in label_sets):
        raise ValueError("Las anotaciones de una misma frase no coinciden en longitud.")

    normalized_sets = [
        [normalize_merge_label(label) for label in labels] for labels in label_sets
    ]

    merged: list[str] = []
    agreements: list[float] = []
    prev = "o"
    for idx in range(length):
        votes = [labels[idx] for labels in normalized_sets]
        if len(set(votes)) == 1:
            candidate = coerce_bio_label(votes[0], prev)
        else:
            candidate = merge_disagreeing_labels(votes, prev)
        merged.append(candidate)
        prev = candidate
        if len(label_sets) == 2:
            raw_a = normalize_merge_label(label_sets[0][idx])
            raw_b = normalize_merge_label(label_sets[1][idx])
            agreements.append(1.0 if raw_a == raw_b else 0.0)
    token_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    return merged, token_agreement


def extract_frase_records(records: list[dict], frase_text: str) -> list[dict] | None:
    """Extract annotation records covering a target sentence substring.

    Args:
        records: Full annotation records from one JSON file.
        frase_text: Lowercased sentence text to locate.

    Returns:
        Slice of records spanning the sentence, or ``None`` if not found.
    """
    full_text, _ = records_to_text_and_labels(records)
    start = full_text.find(frase_text)
    if start < 0:
        norm_full = normalize_annotation_text(full_text)
        norm_frase = normalize_annotation_text(frase_text)
        start = norm_full.find(norm_frase)
        if start < 0:
            return None

    end = start + len(frase_text)
    cursor = 0
    chunk: list[dict] = []
    for record in records:
        piece = record["clave"]
        piece_start = cursor
        piece_end = cursor + len(piece)
        if piece_end <= start:
            cursor = piece_end
            continue
        if piece_start >= end:
            break
        chunk.append(record)
        cursor = piece_end
    rebuilt, _ = records_to_text_and_labels(chunk)
    if rebuilt != frase_text and normalize_annotation_text(
        rebuilt
    ) != normalize_annotation_text(frase_text):
        logger.warning(
            "Segmento parcial no coincide exactamente con la frase objetivo."
        )
    return chunk


def load_assignments(path: Path) -> tuple[str, list[str], list[list[int]]]:
    """Load sentence assignment metadata from ``asignaciones.json``.

    Args:
        path: Path to the assignments JSON file.

    Returns:
        Tuple of granularity label, sentence texts, and per-JSON index lists.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    granularidad = payload.get("granularidad", "palabra")
    return granularidad, payload["frases"], payload["asignaciones"]


def list_annotator_json_paths(json_dir: Path) -> list[Path]:
    """Return sorted paths to ``json_XX.json`` annotator files in a directory."""
    paths = sorted(json_dir.glob("json_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No json_XX.json files found in {json_dir}. Run prepare-annotations first."
        )
    return paths


def resolve_assignments(json_dir: Path) -> tuple[str, list[str], list[list[int]]]:
    """Load or infer which sentences appear in each annotator JSON file.

    Resolution order:

    1. Use ``asignaciones.json`` when present in ``json_dir``.
    2. Otherwise scan ``json_XX.json`` against ``frases_seleccionadas.json``,
       then cache the result as ``asignaciones.json``.

    Args:
        json_dir: Directory with annotator JSON files and metadata.

    Returns:
        Tuple of granularity label, sentence texts, and per-JSON index lists.

    Raises:
        FileNotFoundError: If neither metadata nor annotator JSON files exist.
        ValueError: If ``frases_seleccionadas.json`` has an invalid format.
    """
    json_dir = require_dir(json_dir, label="annotations directory")
    asignaciones_path = json_dir / "asignaciones.json"
    if asignaciones_path.is_file():
        logger.info("Using assignments from {}", asignaciones_path)
        return load_assignments(require_file(asignaciones_path, label="assignments file"))

    frases_path = json_dir / "frases_seleccionadas.json"
    if not frases_path.is_file():
        list_annotator_json_paths(json_dir)
        raise FileNotFoundError(
            f"Missing {asignaciones_path.name} and {frases_path.name} in {json_dir}. "
            "Run prepare-annotations to generate templates, or add asignaciones.json."
        )

    frases = json.loads(
        require_file(frases_path, label="selected sentences file").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(frases, list) or not all(isinstance(f, str) for f in frases):
        raise ValueError(
            f"{frases_path.name} must be a JSON array of sentence strings."
        )

    json_paths = list_annotator_json_paths(json_dir)
    frase_texts = [frase.lower() for frase in frases]
    assignments: list[list[int]] = [[] for _ in range(len(json_paths))]

    for json_idx, json_path in enumerate(json_paths):
        records = json.loads(json_path.read_text(encoding="utf-8"))
        for frase_idx, frase_text in enumerate(frase_texts):
            if extract_frase_records(records, frase_text) is not None:
                assignments[json_idx].append(frase_idx)

    granularidad = "palabra"
    asignaciones_path.write_text(
        json.dumps(
            {
                "granularidad": granularidad,
                "frases": frases,
                "asignaciones": assignments,
                "inferred": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Inferred assignments for {} annotators and {} sentences → {}",
        len(json_paths),
        len(frases),
        asignaciones_path,
    )
    return granularidad, frases, assignments


def merge_annotations(
    json_dir: Path,
    output_path: Path,
) -> MergeBundle:
    """Merge dual annotations from a flat directory into one NER dataset.

    Expects ``json_dir`` with ``json_01.json``, ``json_02.json``, … Assignment
    metadata is loaded from ``asignaciones.json`` or inferred automatically from
    ``frases_seleccionadas.json``.

    Args:
        json_dir: Directory containing per-annotator ``json_XX.json`` files.
        output_path: Destination path for the merged dataset JSON.

    Returns:
        ``MergeBundle`` with global report, merged sentences, and per-sentence details.
    """
    json_dir = require_dir(json_dir, label="annotations directory")
    granularidad, frases, assignments = resolve_assignments(json_dir)
    frase_texts = [frase.lower() for frase in frases]
    logger.info("Merging annotations (granularity={})", granularidad)

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
                "Sentence {} assigned to {} JSON files (expected 2).",
                frase_idx,
                len(json_indices),
            )
            skipped_no_pair += 1
            continue

        word_sets: list[tuple[list[str], list[str]]] = []
        sources: list[str] = []

        for json_idx in json_indices:
            json_path = json_dir / f"json_{json_idx + 1:02d}.json"
            if not json_path.exists():
                logger.warning("Missing {}", json_path.name)
                continue
            records = json.loads(json_path.read_text(encoding="utf-8"))
            chunk = extract_frase_records(records, frase_texts[frase_idx])
            if chunk is None:
                logger.warning("Sentence {} not found in {}", frase_idx, json_path.name)
                continue
            _, tokens, labels = records_to_word_labels(chunk)
            word_sets.append((tokens, labels))
            sources.append(json_path.name)

        if len(word_sets) < 2:
            skipped_unlabeled += 1
            continue

        label_sets = [labels for _, labels in word_sets]
        tokens = word_sets[0][0]
        if word_sets[0][0] != word_sets[1][0]:
            logger.warning(
                "Sentence {}: tokenization differs between annotators; using the first.",
                frase_idx,
            )
        if len(label_sets[0]) != len(label_sets[1]):
            logger.warning(
                "Sentence {} length mismatch: {} vs {} units.",
                frase_idx,
                len(label_sets[0]),
                len(label_sets[1]),
            )
            continue

        norm_a = [normalize_merge_label(label) for label in label_sets[0]]
        norm_b = [normalize_merge_label(label) for label in label_sets[1]]
        kappa = cohen_kappa(norm_a, norm_b)
        merged_labels, agreement = merge_sentence_labels(label_sets)
        disagreements = sum(1 for left, right in zip(norm_a, norm_b) if left != right)

        kappas.append(kappa)
        token_agreements.append(agreement)
        merged_sentences.append(
            {
                "frase_id": frase_idx,
                "text": frase_texts[frase_idx],
                "tokens": tokens,
                "labels": merged_labels,
            }
        )
        frase_details.append(
            FraseMergeResult(
                frase_id=frase_idx,
                text=frase_texts[frase_idx],
                json_indices=json_indices,
                sources=sources,
                label_sets=label_sets,
                merged_labels=merged_labels,
                token_agreement=agreement,
                kappa=kappa,
                disagreements=disagreements,
            )
        )

    report = {
        "source": str(json_dir),
        "granularidad": granularidad,
        "n_frases": len(merged_sentences),
        "skipped_no_pair": skipped_no_pair,
        "skipped_unlabeled": skipped_unlabeled,
        "mean_token_agreement": sum(token_agreements) / max(len(token_agreements), 1),
        "mean_cohen_kappa": sum(kappas) / max(len(kappas), 1),
        "pairwise_kappas": kappas,
    }
    save_merged_dataset(output_path, merged_sentences)
    logger.info("Merged dataset written to {}", output_path)
    logger.info("Mean token agreement: {:.3f}", report["mean_token_agreement"])
    logger.info("Mean Cohen's kappa: {:.3f}", report["mean_cohen_kappa"])

    return MergeBundle(
        report=report, sentences=merged_sentences, frase_details=frase_details
    )
