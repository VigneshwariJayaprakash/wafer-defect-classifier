"""
Wafer Defect Classifier — Production FastAPI Service
=====================================================
Endpoints:
  GET  /health          — liveness + model status
  POST /predict         — single wafer map inference
  POST /predict/batch   — batch inference (up to 100 wafers)
  GET  /classes         — list of defect classes + metadata
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model import classifier
from .schemas import (
    WaferMapRequest, WaferMapResponse, DefectPrediction,
    BatchRequest, BatchResponse,
    HealthResponse
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Lifespan: load models once at startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models...")
    classifier.load()
    logger.info("Models ready.")
    yield
    logger.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wafer Defect Classifier API",
    description=(
        "Production-grade ADC (Automated Defect Classification) service. "
        "Classifies wafer map defect patterns into 8 categories using a "
        "CNN feature extractor + XGBoost/Random Forest ensemble. "
        "Returns defect class, confidence, yield estimate, root cause, and priority."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


# ── Helper ────────────────────────────────────────────────────────────
def _build_response(req: WaferMapRequest, start_time: float) -> WaferMapResponse:
    """Run inference and build response object."""
    result = classifier.predict(req.wafer_map)
    elapsed_ms = (time.time() - start_time) * 1000

    return WaferMapResponse(
        wafer_id=req.wafer_id,
        lot_id=req.lot_id,
        prediction=DefectPrediction(
            defect_class   = result["defect_class"],
            confidence     = result["confidence"],
            yield_estimate = result["yield_estimate"],
            root_cause     = result["root_cause"],
            priority       = result["priority"],
            top3           = result["top3"],
        ),
        defect_density      = result["defect_density"],
        processing_time_ms  = elapsed_ms,
    )


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check + model status."""
    meta = classifier.metadata or {}
    return HealthResponse(
        status          = "ok" if classifier.is_loaded else "loading",
        model_loaded    = classifier.is_loaded,
        model_accuracy  = meta.get("model_accuracy", 0.0),
        model_f1        = meta.get("model_f1_weighted", 0.0),
        defect_classes  = meta.get("defect_classes", []),
        version         = "1.0.0",
    )


@app.post("/predict", response_model=WaferMapResponse, tags=["Inference"])
async def predict(request: WaferMapRequest):
    """
    Classify a single wafer map.

    - **wafer_map**: 2D array of floats (0=good die, 1=failed die, 0.5=untested)
    - **wafer_id**: optional ID for audit logging
    - **lot_id**: optional lot ID

    Returns defect class, confidence score, yield estimate, root cause, and priority.
    """
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    if not request.wafer_map or len(request.wafer_map) < 3:
        raise HTTPException(status_code=422, detail="wafer_map must have at least 3 rows.")

    try:
        start = time.time()
        response = _build_response(request, start)
        logger.info(
            f"Predicted {response.prediction.defect_class} "
            f"(conf={response.prediction.confidence:.3f}, "
            f"priority={response.prediction.priority}) "
            f"for wafer={request.wafer_id}"
        )
        return response
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
async def predict_batch(request: BatchRequest):
    """
    Classify a batch of wafer maps (max 100 per request).

    Returns individual predictions plus batch-level summary
    (critical count, high count).
    """
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    if len(request.wafers) > 100:
        raise HTTPException(
            status_code=422,
            detail="Batch size limit is 100 wafers per request."
        )

    batch_start = time.time()
    results = []
    critical_count = 0
    high_count = 0

    for wafer_req in request.wafers:
        try:
            start = time.time()
            resp = _build_response(wafer_req, start)
            results.append(resp)
            if resp.prediction.priority == "CRITICAL":
                critical_count += 1
            elif resp.prediction.priority == "HIGH":
                high_count += 1
        except Exception as e:
            logger.error(f"Failed on wafer {wafer_req.wafer_id}: {e}")
            raise HTTPException(status_code=500,
                                detail=f"Error on wafer {wafer_req.wafer_id}: {e}")

    total_ms = (time.time() - batch_start) * 1000
    logger.info(
        f"Batch of {len(results)} wafers processed in {total_ms:.1f}ms | "
        f"CRITICAL={critical_count} HIGH={high_count}"
    )

    return BatchResponse(
        results            = results,
        total_wafers       = len(results),
        processing_time_ms = total_ms,
        critical_count     = critical_count,
        high_count         = high_count,
    )


@app.get("/classes", tags=["Metadata"])
async def get_classes():
    """Return all defect classes with root cause and priority metadata."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    meta = classifier.metadata
    return {
        "defect_classes": [
            {
                "class"     : cls,
                "root_cause": meta["root_causes"][cls],
                "priority"  : meta["priority"][cls],
            }
            for cls in meta["defect_classes"]
        ]
    }
