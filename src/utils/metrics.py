from typing import cast

import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(
        y_true,
        y_pred,
        metric_names: list[str],
        y_proba=None,
        ) -> dict[str, float | None]:

    results = {}

    for metric_name in metric_names:
        if metric_name == "accuracy":
            results["accuracy"] = accuracy_score(y_true, y_pred)
        elif metric_name == "precision":
            results["precision"] = precision_score(y_true, y_pred, zero_division=0)

        elif metric_name == "recall":
            results["recall"] = recall_score(y_true, y_pred, zero_division=0)

        elif metric_name == "f1":
            results["f1"] = f1_score(y_true, y_pred, zero_division=0)

        elif metric_name == "roc_auc":
            if y_proba is None:
                results["roc_auc"] = None
            else:
                results["roc_auc"] = roc_auc_score(y_true, y_proba)

        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return results

def summarize_metrics(fold_metrics_df: pl.DataFrame, metric_names: list[str]) -> dict[str, float]:
    summary = {}
    for metric_name in metric_names:
        values = fold_metrics_df.get_column(metric_name).drop_nulls()
        mean_value = cast(float | None, values.mean())
        std_value = cast(float | None, values.std())

        if mean_value is None:
            raise ValueError(f"Cannot summarize metric with no values: {metric_name}")

        summary[f"mean_{metric_name}"] = float(mean_value)
        summary[f"std_{metric_name}"] = float(std_value or 0.0)
    return summary
