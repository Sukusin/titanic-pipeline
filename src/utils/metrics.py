from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import polars as pl


def calculate_metrics(
        y_true,
        y_pred,
        metric_names: list[str],
        y_proba=None,
        ) -> dict[str, float]:

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
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())
    return summary
