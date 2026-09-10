"""
SHAP-based explainability for the ANN production model.
Computes per-prediction feature attributions, answering "why did the model
predict this move?" — feeds Phase 21's Psychology Dashboard and the
predictions.explanation_json column (Phase 3 schema).
"""

import json

import pandas as pd
import shap
import torch

from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline
from neuroplay.models.torch_dataset import FEATURE_COLUMNS

logger = get_logger(__name__)


class ANNExplainer:
    """Wraps SHAP's GradientExplainer around the trained ANN for feature attribution."""

    def __init__(self, ann: ANNBaseline, background_df: pd.DataFrame, sample_size: int = 100):
        """
        background_df: a representative sample of (scaled) training features,
        used as the SHAP baseline distribution.
        """
        self.ann = ann
        background_sample = background_df[FEATURE_COLUMNS].sample(
            n=min(sample_size, len(background_df)), random_state=42
        )
        background_tensor = torch.tensor(background_sample.to_numpy(), dtype=torch.float32)
        self.explainer = shap.GradientExplainer(self.ann.model, background_tensor)

    def explain(self, df: pd.DataFrame) -> list[dict]:
        """
        Returns a list of explanation dicts, one per row in df:
        {"predicted_move": int, "feature_attributions": {feature_name: shap_value}}
        """
        if self.ann.scaler is not None:
            from neuroplay.models.feature_scaler import apply_scaler

            df = apply_scaler(df, self.ann.scaler)

        features_tensor = torch.tensor(df[FEATURE_COLUMNS].to_numpy(), dtype=torch.float32)
        shap_values = self.explainer.shap_values(features_tensor)
        # shap_values shape: (num_samples, num_features, num_classes)

        predictions = self.ann.model(features_tensor).argmax(dim=1).detach().numpy()

        explanations = []
        for i, pred in enumerate(predictions):
            attributions = {
                FEATURE_COLUMNS[j]: round(float(shap_values[i, j, pred]), 4)
                for j in range(len(FEATURE_COLUMNS))
            }
            explanations.append({"predicted_move": int(pred), "feature_attributions": attributions})
        return explanations

    def explain_to_json(self, df: pd.DataFrame) -> list[str]:
        """
        Returns explanations as JSON strings, ready for the
        predictions.explanation_json column.
        """
        explanations = self.explain(df)
        return [json.dumps(e) for e in explanations]
