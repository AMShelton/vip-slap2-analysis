from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


def _normalize_quality(x: object) -> str:
    if pd.isna(x):
        return "unknown"

    q = str(x).strip().lower()

    if q in {"good", "great", "excellent", "pass"}:
        return "good"
    if q in {"ok", "okay", "acceptable", "usable", "medium", "fair"}:
        return "okay"
    if q in {"poor", "bad", "fail", "failed"}:
        return "poor"

    return q


def _clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _overall_mouse_assessment(mouse_df: pd.DataFrame) -> str:
    counts = mouse_df["quality_simple"].value_counts(dropna=False).to_dict()

    n_good = counts.get("good", 0)
    n_okay = counts.get("okay", 0)
    n_poor = counts.get("poor", 0)
    n_unknown = counts.get("unknown", 0)
    n_total = len(mouse_df)

    if n_total == 0:
        return "no sessions"

    good_frac = n_good / n_total

    if n_good >= 2 and n_poor == 0 and good_frac >= 0.6:
        return "strong"
    if (n_good + n_okay) >= max(1, int(0.75 * n_total)):
        return "usable"
    if n_poor >= max(1, n_total // 2):
        return "needs review"
    if n_unknown == n_total:
        return "unclear"

    return "mixed"


def build_dataset_quality_overview(
    dataset_root: str | Path,
    session_summary_path: str | Path,
    mouse_ids: Sequence[int | str],
    output_csv_name: str = "dataset_quality_overview.csv",
    output_md_name: str = "DATASET_QUALITY_OVERVIEW.md",
    include_expression_checks: bool = False,
    paradigm_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a lightweight top-level dataset quality overview for designated mice.

    Parameters
    ----------
    dataset_root
        Root dataset directory where output files will be written.
    session_summary_path
        Path to VIP_SD_summary.xlsx.
    mouse_ids
        Mouse / subject IDs to include.
    output_csv_name
        Name of CSV output file written to dataset_root.
    output_md_name
        Name of markdown output file written to dataset_root.
    include_expression_checks
        Whether to keep expression_check sessions.
    paradigm_filter
        Optional exact-match filter for paradigm, e.g. "change_detection_passive".

    Returns
    -------
    pd.DataFrame
        Session-level filtered dataframe used to generate outputs.
    """
    dataset_root = Path(dataset_root)
    session_summary_path = Path(session_summary_path)

    mouse_ids_norm = {str(m).strip() for m in mouse_ids}

    sessions = pd.read_excel(session_summary_path, sheet_name="sessions").copy()

    # Normalize key columns
    sessions["subject_id"] = sessions["subject_id"].astype("Int64").astype(str)
    sessions["session_date"] = pd.to_datetime(sessions["session_date"], errors="coerce")
    sessions["quality_simple"] = sessions["quality"].map(_normalize_quality)
    sessions["flags_clean"] = sessions["flags"].map(_clean_text)
    sessions["notes_clean"] = sessions["notes"].map(_clean_text)
    sessions["indicator1_clean"] = sessions["indicator1"].map(_clean_text)
    sessions["indicator2_clean"] = sessions["indicator2"].map(_clean_text)
    sessions["session_type_clean"] = sessions["session_type"].map(_clean_text)
    sessions["paradigm_clean"] = sessions["paradigm"].map(_clean_text)

    # Filter to designated mice
    sessions = sessions.loc[sessions["subject_id"].isin(mouse_ids_norm)].copy()

    if paradigm_filter is not None:
        sessions = sessions.loc[sessions["paradigm_clean"] == paradigm_filter].copy()

    if not include_expression_checks:
        sessions = sessions.loc[sessions["session_type_clean"].str.lower() != "expression_check"].copy()

    # Keep only lightweight, human-useful columns
    session_view = sessions[
        [
            "subject_id",
            "session_id",
            "session_date",
            "session_#",
            "indicator1_clean",
            "indicator2_clean",
            "session_type_clean",
            "stimulus",
            "quality",
            "quality_simple",
            "flags_clean",
            "notes_clean",
            "session_dir",
        ]
    ].copy()

    session_view = session_view.rename(
        columns={
            "subject_id": "mouse_id",
            "session_#": "session_number",
            "indicator1_clean": "indicator1",
            "indicator2_clean": "indicator2",
            "session_type_clean": "session_type",
            "flags_clean": "flags",
            "notes_clean": "notes",
        }
    )

    session_view = session_view.sort_values(
        ["mouse_id", "session_date", "session_number", "session_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    # Mouse-level rollup
    mouse_rows = []
    for mouse_id, mouse_df in session_view.groupby("mouse_id", sort=True):
        latest_date = mouse_df["session_date"].max()
        latest_date_str = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else ""

        indicators = sorted(
            {
                x for x in pd.concat([mouse_df["indicator1"], mouse_df["indicator2"]]).tolist()
                if isinstance(x, str) and x.strip()
            }
        )

        quality_counts = mouse_df["quality_simple"].value_counts().to_dict()

        mouse_rows.append(
            {
                "mouse_id": mouse_id,
                "n_sessions": len(mouse_df),
                "n_good": quality_counts.get("good", 0),
                "n_okay": quality_counts.get("okay", 0),
                "n_poor": quality_counts.get("poor", 0),
                "n_unknown": quality_counts.get("unknown", 0),
                "latest_session_date": latest_date_str,
                "indicators": ", ".join(indicators),
                "overall_assessment": _overall_mouse_assessment(mouse_df),
            }
        )

    mouse_summary = pd.DataFrame(mouse_rows).sort_values("mouse_id").reset_index(drop=True)

    # Write CSV (session rows only; easiest to sort/filter later)
    csv_path = dataset_root / output_csv_name
    session_view.to_csv(csv_path, index=False)

    # Write lightweight markdown overview
    md_path = dataset_root / output_md_name
    lines = []
    lines.append("# Dataset Quality Overview")
    lines.append("")
    lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset root: `{dataset_root}`")
    lines.append("")
    lines.append("## Included mice")
    lines.append("")
    lines.append(", ".join(str(m) for m in mouse_ids))
    lines.append("")
    lines.append("## Mouse summaries")
    lines.append("")

    if mouse_summary.empty:
        lines.append("_No matching sessions found._")
    else:
        for _, row in mouse_summary.iterrows():
            lines.append(f"### Mouse {row['mouse_id']}")
            lines.append("")
            lines.append(f"- Overall assessment: **{row['overall_assessment']}**")
            lines.append(f"- Sessions: {row['n_sessions']}")
            lines.append(
                f"- Quality counts: good={row['n_good']}, okay={row['n_okay']}, poor={row['n_poor']}, unknown={row['n_unknown']}"
            )
            lines.append(f"- Latest session: {row['latest_session_date']}")
            lines.append(f"- Indicators: {row['indicators'] if row['indicators'] else 'n/a'}")
            lines.append("")

            mouse_sessions = session_view.loc[session_view["mouse_id"] == row["mouse_id"]].copy()

            for _, srow in mouse_sessions.iterrows():
                date_str = srow["session_date"].strftime("%Y-%m-%d") if pd.notna(srow["session_date"]) else "unknown-date"

                brief = f"- {date_str} | {srow['session_type']} | quality={srow['quality']}"
                if srow["flags"]:
                    brief += f" | flags={srow['flags']}"
                if srow["notes"]:
                    brief += f" | notes={srow['notes']}"
                lines.append(brief)

            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    return session_view