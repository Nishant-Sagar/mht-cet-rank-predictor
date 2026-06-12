from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass(frozen=True)
class PredictorConfig:
    total_candidates: int = 300000
    rank_buffer: int = 5000
    medium_gap: int = 2000


class MAHCETCollegePredictor:
    REQUIRED_COLS = {"college", "branch", "category", "closing_rank", "closing_percentile"}
    CATEGORY_ALIASES = {
        "OPEN": {"OPEN", "GOPEN", "LOPEN"},
        "OBC": {"OBC", "GOBC", "LOBC", "SEBC", "GSEBC", "LSEBC"},
        "SC": {"SC", "GSC", "LSC"},
        "ST": {"ST", "GST", "LST"},
        "NT1": {"NT1", "NTA", "GNTA", "LNTA"},
        "NT2": {"NT2", "NTB", "GNTB", "LNTB"},
        "NT3": {"NT3", "NTC", "GNTC", "LNTC"},
        "VJ": {"VJ", "NTD", "GNTD", "LNTD"},
        "EWS": {"EWS"},
        "TFWS": {"TFWS"},
    }

    def __init__(self, csv_path: str, config: PredictorConfig | None = None):
        self.csv_path = csv_path
        self.config = config or PredictorConfig()
        self.df = self._load_and_normalize(csv_path)

    def _load_and_normalize(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        df = df.copy()
        df["college"] = df["college"].astype(str).fillna("").str.strip()
        df["branch"] = df["branch"].astype(str).fillna("").str.strip()
        df["category"] = df["category"].astype(str).fillna("").str.strip().str.upper()

        df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
        df["closing_percentile"] = pd.to_numeric(df["closing_percentile"], errors="coerce")

        df = df.dropna(subset=["college", "branch", "category", "closing_rank", "closing_percentile"])
        df = df[df["college"] != ""]
        df = df[df["branch"] != ""]
        df = df[df["category"] != ""]

        df["closing_rank"] = df["closing_rank"].astype(int)
        df["closing_percentile"] = df["closing_percentile"].astype(float)

        return df

    def percentile_to_rank(self, percentile: float) -> int:
        rank = int((100.0 - float(percentile)) * self.config.total_candidates / 100.0)
        return max(rank, 1)

    _MARKS_PERCENTILE_BREAKPOINTS = [
        (0,   0.0),
        (50,  10.0),
        (80,  30.0),
        (100, 50.0),
        (120, 70.0),
        (140, 85.0),
        (155, 93.0),
        (165, 97.0),
        (175, 99.0),
        (185, 99.5),
        (200, 100.0),
    ]

    def marks_to_percentile(self, marks: float) -> float:
        m = max(0.0, min(200.0, float(marks)))
        pts = self._MARKS_PERCENTILE_BREAKPOINTS
        for i in range(len(pts) - 1):
            m0, p0 = pts[i]
            m1, p1 = pts[i + 1]
            if m0 <= m <= m1:
                t = (m - m0) / (m1 - m0)
                return round(p0 + t * (p1 - p0), 4)
        return 100.0

    def predict(
        self,
        *,
        user_rank: int,
        category: str,
        branch: Optional[str] = None,
        top_n: int = 15,
    ) -> List[Dict[str, Any]]:
        if user_rank is None or user_rank <= 0:
            raise ValueError("user_rank must be a positive integer")

        category = (category or "").strip().upper()
        if not category:
            raise ValueError("category is required")

        if top_n <= 0:
            top_n = 15
        top_n = min(top_n, 50)

        df = self.df

        valid_categories = list(self.CATEGORY_ALIASES.keys())
        category_codes = self.CATEGORY_ALIASES.get(category)
        if category_codes is None:
            raise ValueError(
                f"Invalid category '{category}'. Valid options: {', '.join(valid_categories)}"
            )
        df = df[df["category"].isin(category_codes)]
        if df.empty:
            return []

        if branch:
            b = branch.strip()
            if b:
                words = b.lower().split()
                mask = df["branch"].apply(
                    lambda x: all(w in x.lower() for w in words)
                )
                df_branch = df[mask]
                if df_branch.empty:
                    available = sorted(df["branch"].unique())
                    raise ValueError(
                        f"No branches matching '{b}' found for category {category}. "
                        f"Available branches: {', '.join(available[:20])}"
                        + (" ..." if len(available) > 20 else "")
                    )
                df = df_branch

        df = df.copy()
        df["rank_gap"] = df["closing_rank"] - int(user_rank)

        df_in_range = df[df["rank_gap"] >= -self.config.rank_buffer]
        if df_in_range.empty:
            df = df.nlargest(top_n * 4, "closing_rank")
        else:
            df = df_in_range

        def chance_label(gap: int) -> str:
            if gap >= 0:
                return "High"
            if gap >= -self.config.medium_gap:
                return "Medium"
            return "Low"

        df["chance"] = df["rank_gap"].apply(chance_label)

        chance_priority = {"High": 1, "Medium": 2, "Low": 3}
        df["priority"] = df["chance"].map(chance_priority).fillna(3).astype(int)
        df["eligible"] = df["rank_gap"] >= 0

        branch_filter_active = bool(branch and branch.strip())

        def agg_branches(x: pd.Series) -> list:
            if branch_filter_active:
                return sorted(set(x))
            eligible_mask = df.loc[x.index, "eligible"]
            eligible = sorted(set(x[eligible_mask]))
            return eligible if eligible else sorted(set(x))

        def best_branch_for_group(group: pd.DataFrame) -> str:
            eligible = group[group["eligible"]]
            if not eligible.empty:
                # most competitive branch the student still qualifies for
                return eligible.loc[eligible["closing_rank"].idxmin(), "branch"]
            # fallback: most accessible branch in the college
            return group.loc[group["closing_rank"].idxmax(), "branch"]

        best_branch_map = (
            df.groupby("college", group_keys=False)
            .apply(best_branch_for_group)
            .rename("best_branch")
        )

        grouped = (
            df.groupby("college", as_index=False)
            .agg(
                branches=("branch", agg_branches),
                best_closing_rank=("closing_rank", "min"),
                best_closing_percentile=("closing_percentile", "max"),
                priority=("priority", "min"),
                chance=("chance", lambda x: sorted(set(x), key=lambda c: chance_priority.get(c, 9))[0]),
            )
        )

        grouped = grouped.merge(best_branch_map, on="college")

        grouped = grouped.sort_values(
            by=["priority", "best_closing_rank"],
            ascending=[True, True],
        ).head(top_n)

        return grouped.to_dict(orient="records")
