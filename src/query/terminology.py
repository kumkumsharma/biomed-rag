"""Deterministic biomedical terminology normalization.

The original query is never discarded. Abbreviations and common synonyms are
detected with a local mapping; no LLM or network call is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Bidirectional biomedical terms useful for BioASQ-style retrieval.
# Keys are lowercase surface forms.
TERM_MAP: dict[str, tuple[str, ...]] = {
    "egfr": ("epidermal growth factor receptor",),
    "epidermal growth factor receptor": ("EGFR",),
    "tnf": ("tumor necrosis factor",),
    "tumor necrosis factor": ("TNF",),
    "il-6": ("interleukin-6", "interleukin 6"),
    "il6": ("interleukin-6", "interleukin 6"),
    "interleukin-6": ("IL-6", "IL6"),
    "interleukin 6": ("IL-6", "IL6"),
    "mi": ("myocardial infarction",),
    "myocardial infarction": ("MI", "heart attack"),
    "heart attack": ("myocardial infarction", "MI"),
    "copd": ("chronic obstructive pulmonary disease",),
    "chronic obstructive pulmonary disease": ("COPD",),
    "ada": ("adenosine deaminase",),
    "adenosine deaminase": ("ADA",),
    "hiv": ("human immunodeficiency virus",),
    "human immunodeficiency virus": ("HIV",),
    "hcv": ("hepatitis C virus",),
    "hepatitis c virus": ("HCV",),
    "hbv": ("hepatitis B virus",),
    "hepatitis b virus": ("HBV",),
    "nsclc": ("non-small cell lung cancer",),
    "non-small cell lung cancer": ("NSCLC",),
    "sclc": ("small cell lung cancer",),
    "t2dm": ("type 2 diabetes mellitus",),
    "type 2 diabetes": ("T2DM", "type 2 diabetes mellitus"),
    "diabetes mellitus": ("diabetes",),
    "ace": ("angiotensin converting enzyme",),
    "angiotensin converting enzyme": ("ACE",),
    "ace inhibitor": ("angiotensin converting enzyme inhibitor",),
    "vegf": ("vascular endothelial growth factor",),
    "vascular endothelial growth factor": ("VEGF",),
    "her2": ("human epidermal growth factor receptor 2", "ERBB2"),
    "erbb2": ("HER2",),
    "brca1": ("breast cancer gene 1",),
    "brca2": ("breast cancer gene 2",),
    "p53": ("TP53",),
    "tp53": ("p53",),
    "pten": ("phosphatase and tensin homolog",),
    "mapk": ("mitogen-activated protein kinase",),
    "pi3k": ("phosphoinositide 3-kinase",),
    "mrna": ("messenger RNA",),
    "dna": ("deoxyribonucleic acid",),
    "rna": ("ribonucleic acid",),
    "pcr": ("polymerase chain reaction",),
    "polymerase chain reaction": ("PCR",),
    "snps": ("single nucleotide polymorphisms",),
    "snp": ("single nucleotide polymorphism",),
    "gwas": ("genome-wide association study",),
    "als": ("amyotrophic lateral sclerosis",),
    "amyotrophic lateral sclerosis": ("ALS",),
    "ad": ("Alzheimer's disease",),
    "alzheimer": ("Alzheimer's disease",),
    "alzheimer's disease": ("AD",),
    "pd": ("Parkinson's disease",),
    "parkinson": ("Parkinson's disease",),
    "ms": ("multiple sclerosis",),
    "multiple sclerosis": ("MS",),
    "ra": ("rheumatoid arthritis",),
    "rheumatoid arthritis": ("RA",),
    "ibd": ("inflammatory bowel disease",),
    "inflammatory bowel disease": ("IBD",),
    "uc": ("ulcerative colitis",),
    "cd": ("Crohn's disease",),
    "covid-19": ("SARS-CoV-2", "coronavirus disease 2019"),
    "sars-cov-2": ("COVID-19",),
    "ace2": ("angiotensin converting enzyme 2",),
    "hirschsprung": ("Hirschsprung disease", "congenital aganglionic megacolon"),
    "hirschsprung disease": ("congenital aganglionic megacolon", "Hirschsprung"),
    "papilin": ("Ppn",),
    "insulin": ("INS",),
    "egf": ("epidermal growth factor",),
    "epidermal growth factor": ("EGF",),
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'+-]*")


@dataclass(frozen=True)
class TerminologyResult:
    original: str
    normalized: str
    detected_terms: tuple[str, ...]
    mappings: tuple[tuple[str, str], ...]
    method: str = "local_term_map"

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "detected_terms": list(self.detected_terms),
            "mappings": [{"from": src, "to": dst} for src, dst in self.mappings],
            "method": self.method,
        }


def _lookup_key(token: str) -> str:
    return token.strip().lower()


def detect_terms(query: str) -> list[str]:
    if not isinstance(query, str) or not query.strip():
        return []
    found: list[str] = []
    lowered = query.lower()
    for surface in sorted(TERM_MAP, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])")
        if pattern.search(lowered) and surface not in found:
            found.append(surface)
    return found


def normalize_query(query: str, *, enabled: bool = True) -> TerminologyResult:
    """Expand detected abbreviations into a parallel normalized query.

    The original string is returned unchanged in ``original``.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    original = query
    if not enabled or not query.strip():
        return TerminologyResult(original=original, normalized=original, detected_terms=(), mappings=())

    detected = tuple(detect_terms(query))
    mappings: list[tuple[str, str]] = []
    normalized = query
    for surface in detected:
        expansions = TERM_MAP.get(surface) or ()
        if not expansions:
            continue
        primary = expansions[0]
        mappings.append((surface, primary))
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])", re.IGNORECASE)
        if pattern.search(normalized) and primary.lower() not in normalized.lower():
            normalized = pattern.sub(lambda _match, text=primary: text, normalized, count=1)
    if normalized.strip() == original.strip():
        extra = []
        for surface in detected:
            for variant in TERM_MAP.get(surface) or ():
                if variant.lower() not in original.lower():
                    extra.append(variant)
                    break
        if extra:
            normalized = original.rstrip() + " " + " ".join(extra[:3])
            for surface, variant in zip(detected, extra[:3]):
                if (surface, variant) not in mappings:
                    mappings.append((surface, variant))
    return TerminologyResult(
        original=original,
        normalized=normalized,
        detected_terms=detected,
        mappings=tuple(mappings),
    )
