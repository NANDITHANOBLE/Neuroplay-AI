"""
CLI entry point — generates SHAP explanations for a sample of test predictions,
saves per-prediction and global importance visualizations.
Run: python -m neuroplay.explainability.run_explainability
"""

import pandas as pd

from neuroplay.config import settings
from neuroplay.explainability.plot_explanations import (
    plot_global_feature_importance,
    plot_single_explanation,
)
from neuroplay.explainability.shap_explainer import ANNExplainer
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline

logger = get_logger(__name__)

SAMPLE_SIZE = 50


def main() -> None:
    processed_dir = settings.data_dir / "processed"
    features_df = pd.read_parquet(processed_dir / "features_full.parquet")
    train_df = pd.read_parquet(processed_dir / "train.parquet")

    train_features = features_df[features_df["match_id"].isin(train_df["match_id"].unique())]
    sample_df = features_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    device = "cpu"
    ann_dir = settings.models_dir / "ann"
    ann = ANNBaseline(device=device)
    ann.load_weights(str(ann_dir / "ann_best.pt"))
    ann.load_scaler(str(ann_dir / "feature_scaler.json"))

    logger.info(
        f"Building SHAP explainer with background sample from "
        f"{len(train_features)} training rows..."
    )

    from neuroplay.models.feature_scaler import apply_scaler

    scaled_train_features = apply_scaler(train_features, ann.scaler)
    explainer = ANNExplainer(ann, scaled_train_features)

    logger.info(f"Generating explanations for {SAMPLE_SIZE} sample predictions...")
    explanations = explainer.explain(sample_df)

    eval_dir = settings.base_dir / "assets" / "explainability"
    plot_single_explanation(explanations[0], eval_dir / "single_explanation_example.png")
    plot_global_feature_importance(explanations, eval_dir / "global_feature_importance.png")

    logger.info(f"✅ Explainability complete. {len(explanations)} explanations generated.")
    logger.info(f"Example explanation: {explanations[0]}")


if __name__ == "__main__":
    main()
