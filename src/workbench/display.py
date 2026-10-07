"""Display helpers. Missing measurements stay 'Not measured'."""

from __future__ import annotations

from typing import Any


def not_measured(value: Any, *, kind: str = "raw") -> str:
    if value is None:
        return "Not measured"
    if kind == "pct":
        return f"{float(value) * 100:.2f}%"
    if kind == "pp":
        return f"{float(value) * 100:+.2f} pp"
    if kind == "ms":
        return f"{float(value):,.1f} ms"
    if kind == "usd":
        return f"${float(value):.6f}"
    if kind == "ratio":
        return f"{float(value):.4f}"
    return str(value)


def display_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace None with Not measured for UI tables."""

    out = []
    for row in rows:
        out.append({key: ("Not measured" if value is None else value) for key, value in row.items()})
    return out


_PCT_COLUMNS = {
    "Candidate Recall@50",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Recall@10",
    "Rerank Recovery",
}
_MS_COLUMNS = {"Mean latency", "P50 latency", "P95 latency"}
_RATIO_COLUMNS = {"MRR@10", "nDCG@10"}


def format_leaderboard_ui(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key == "id":
                continue
            if value is None:
                item[key] = "Not measured"
            elif key in _PCT_COLUMNS:
                item[key] = not_measured(value, kind="pct")
            elif key in _MS_COLUMNS:
                item[key] = not_measured(value, kind="ms")
            elif key in _RATIO_COLUMNS:
                item[key] = not_measured(value, kind="ratio")
            elif key == "Cost/query":
                item[key] = not_measured(value, kind="usd")
            else:
                item[key] = value
        formatted.append(item)
    return formatted
