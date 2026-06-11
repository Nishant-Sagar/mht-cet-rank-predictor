from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from predictor import MAHCETCollegePredictor, PredictorConfig

CSV_PATH = os.getenv(
    "CSV_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mahcet_cutoff.csv"),
)

predictor: MAHCETCollegePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = MAHCETCollegePredictor(CSV_PATH)
    yield


app = FastAPI(
    title="MHT-CET College Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── response schema ────────────────────────────────────────────────────────────

class CollegePrediction(BaseModel):
    college: str
    best_branch: str
    branches: List[str]
    best_closing_rank: int
    best_closing_percentile: float
    chance: str  # "High" | "Medium" | "Low"


class PredictResponse(BaseModel):
    user_rank: int
    category: str
    branch_filter: Optional[str]
    results: List[CollegePrediction]


class ConvertResponse(BaseModel):
    marks: Optional[float] = None
    percentile: float
    estimated_rank: int


# ── helpers ───────────────────────────────────────────────────────────────────

_NULL_STRINGS = {"null", "none", "undefined", ""}

def normalize_branch(branch: Optional[str]) -> Optional[str]:
    if branch is None:
        return None
    cleaned = branch.strip().lower()
    return None if cleaned in _NULL_STRINGS else branch.strip()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/predict", response_model=PredictResponse, summary="Predict colleges by rank")
def predict_by_rank(
    rank: int = Query(..., gt=0, description="Your MHT-CET rank"),
    category: str = Query(..., description="Reservation category (e.g. OPEN, OBC, SC, ST, EWS, NT1, NT2, NT3, VJ)"),
    branch: Optional[str] = Query(None, description="Branch keyword filter (e.g. 'Computer', 'Mechanical')"),
    top_n: int = Query(15, ge=1, le=50, description="Number of results to return"),
):
    branch = normalize_branch(branch)
    try:
        results = predictor.predict(
            user_rank=rank,
            category=category,
            branch=branch,
            top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        user_rank=rank,
        category=category.upper(),
        branch_filter=branch,
        results=results,
    )


@app.get("/predict/by-percentile", response_model=PredictResponse, summary="Predict colleges by percentile")
def predict_by_percentile(
    percentile: float = Query(..., ge=0.0, le=100.0, description="Your MHT-CET percentile"),
    category: str = Query(..., description="Reservation category"),
    branch: Optional[str] = Query(None, description="Branch keyword filter"),
    top_n: int = Query(15, ge=1, le=50),
):
    branch = normalize_branch(branch)
    rank = predictor.percentile_to_rank(percentile)
    try:
        results = predictor.predict(
            user_rank=rank,
            category=category,
            branch=branch,
            top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        user_rank=rank,
        category=category.upper(),
        branch_filter=branch,
        results=results,
    )


@app.get("/predict/by-marks", response_model=PredictResponse, summary="Predict colleges by marks")
def predict_by_marks(
    marks: float = Query(..., ge=0.0, le=200.0, description="Your MHT-CET marks (0–200)"),
    category: str = Query(..., description="Reservation category"),
    branch: Optional[str] = Query(None, description="Branch keyword filter"),
    top_n: int = Query(15, ge=1, le=50),
):
    branch = normalize_branch(branch)
    percentile = predictor.marks_to_percentile(marks)
    rank = predictor.percentile_to_rank(percentile)
    try:
        results = predictor.predict(
            user_rank=rank,
            category=category,
            branch=branch,
            top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        user_rank=rank,
        category=category.upper(),
        branch_filter=branch,
        results=results,
    )


@app.get("/convert", response_model=ConvertResponse, summary="Convert marks/percentile to rank")
def convert(
    marks: Optional[float] = Query(None, ge=0.0, le=200.0, description="Marks (0–200)"),
    percentile: Optional[float] = Query(None, ge=0.0, le=100.0, description="Percentile (0–100)"),
):
    if marks is None and percentile is None:
        raise HTTPException(status_code=422, detail="Provide either 'marks' or 'percentile'.")
    if marks is not None and percentile is not None:
        raise HTTPException(status_code=422, detail="Provide only one of 'marks' or 'percentile', not both.")

    if marks is not None:
        pct = predictor.marks_to_percentile(marks)
    else:
        pct = float(percentile)

    rank = predictor.percentile_to_rank(pct)
    return ConvertResponse(marks=marks, percentile=round(pct, 4), estimated_rank=rank)


@app.get("/categories", summary="List valid reservation categories")
def list_categories():
    return {"categories": sorted(MAHCETCollegePredictor.CATEGORY_ALIASES.keys())}


@app.get("/branches", summary="List available branches (optionally filtered by category)")
def list_branches(
    category: Optional[str] = Query(None, description="Filter branches by category"),
):
    df = predictor.df
    if category:
        cat = category.strip().upper()
        codes = MAHCETCollegePredictor.CATEGORY_ALIASES.get(cat)
        if codes is None:
            raise HTTPException(status_code=422, detail=f"Unknown category '{category}'.")
        df = df[df["category"].isin(codes)]

    branches = sorted(df["branch"].unique().tolist())
    return {"branches": branches}


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": len(predictor.df)}
