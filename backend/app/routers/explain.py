"""
Explainability endpoint — returns SHAP feature attributions for a given
prediction (identified by move_id).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from neuroplay.db.models import Prediction

from ..dependencies import get_db_session
from ..schemas import ExplanationResponse

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("/{move_id}", response_model=ExplanationResponse)
def explain_prediction(move_id: int, db: Session = Depends(get_db_session)):
    prediction = db.query(Prediction).filter_by(move_id=move_id).first()
    if prediction is None or prediction.explanation_json is None:
        raise HTTPException(status_code=404, detail="No explanation found for this move_id.")

    import json

    explanation = json.loads(prediction.explanation_json)
    return ExplanationResponse(
        move_id=move_id,
        predicted_move=explanation["predicted_move"],
        feature_attributions=explanation["feature_attributions"],
    )
