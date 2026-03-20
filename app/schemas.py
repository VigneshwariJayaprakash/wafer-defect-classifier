"""
Request and response schemas for the Wafer Defect Classifier API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class WaferMapRequest(BaseModel):
    """Single wafer map inference request."""
    wafer_map: List[List[float]] = Field(
        ...,
        description="2D wafer map as nested list of floats (0=pass, 1=fail, 0.5=untested)"
    )
    wafer_id: Optional[str] = Field(
        default=None,
        description="Optional wafer identifier for audit logging"
    )
    lot_id: Optional[str] = Field(
        default=None,
        description="Optional lot identifier"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "wafer_id": "LOT001-W03",
                "lot_id": "LOT001",
                "wafer_map": [[0, 0, 1], [0, 1, 1], [0, 0, 0]]
            }
        }


class DefectPrediction(BaseModel):
    """Single defect prediction result."""
    defect_class: str
    confidence: float
    yield_estimate: float
    root_cause: str
    priority: str
    top3: List[dict]


class WaferMapResponse(BaseModel):
    """Full response for a single wafer map."""
    wafer_id: Optional[str]
    lot_id: Optional[str]
    prediction: DefectPrediction
    defect_density: float
    processing_time_ms: float
    model_version: str = "1.0.0"


class BatchRequest(BaseModel):
    """Batch inference request for multiple wafer maps."""
    wafers: List[WaferMapRequest]

    class Config:
        json_schema_extra = {
            "example": {
                "wafers": [
                    {"wafer_id": "W01", "wafer_map": [[0, 1], [1, 0]]},
                    {"wafer_id": "W02", "wafer_map": [[0, 0], [0, 1]]}
                ]
            }
        }


class BatchResponse(BaseModel):
    """Batch inference response."""
    results: List[WaferMapResponse]
    total_wafers: int
    processing_time_ms: float
    critical_count: int
    high_count: int


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    model_accuracy: float
    model_f1: float
    defect_classes: List[str]
    version: str
