"""
Utility functions for churn prediction analysis.
Contains plotting and evaluation helpers for threshold analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_threshold_analysis(y_true, y_probs, thresholds_to_mark=None):
    """
    Create ROC curve and Precision-Recall plots with marked thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_probs : array-like
        Predicted probabilities for the positive class
    thresholds_to_mark : list, optional
        List of threshold values to mark on the plots (default: [0.30, 0.50, 0.60, 0.75])

    Returns
    -------
    dict
        Dictionary with 'recall' and 'fpr' values at each threshold
    """
    if thresholds_to_mark is None:
        thresholds_to_mark = [0.30, 0.50, 0.60, 0.75]

    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: ROC Curve
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')

    # Mark specific thresholds on ROC curve
    for threshold in thresholds_to_mark:
        idx = np.argmin(np.abs(roc_thresholds - threshold))
        axes[0].scatter(fpr[idx], tpr[idx], s=100, zorder=5)
        axes[0].annotate(
            f'{threshold:.2f}',
            xy=(fpr[idx], tpr[idx]),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0].set_title('ROC Curve with Threshold Markers', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Precision-Recall Curve
    axes[1].plot(recall, precision, linewidth=2, label='Precision-Recall curve')

    # Mark specific thresholds on Precision-Recall curve
    for threshold in thresholds_to_mark:
        idx = np.argmin(np.abs(pr_thresholds - threshold))
        axes[1].scatter(recall[idx], precision[idx], s=100, zorder=5)
        axes[1].annotate(
            f'{threshold:.2f}',
            xy=(recall[idx], precision[idx]),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve with Threshold Markers', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate and print metrics at each threshold
    recall_fpr = print_metrics_table(y_true, y_probs, thresholds_to_mark)

    return recall_fpr


def print_metrics_table(y_true, y_probs, thresholds):
    """
    Print a formatted table of metrics at different thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_probs : array-like
        Predicted probabilities
    thresholds : list
        Threshold values to evaluate

    Returns
    -------
    dict
        Dictionary with 'recall' and 'fpr' lists
    """
    print("\n" + "=" * 70)
    print("Metrics at Different Thresholds")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'FPR':<10} {'F1':<10}")
    print("-" * 70)

    recall_fpr = {"recall": [], "fpr": []}

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

        recall_fpr["recall"].append(recall_val)
        recall_fpr["fpr"].append(fpr_val)

        print(f"{threshold:<12.2f} {recall_val:<10.3f} {precision_val:<12.3f} {fpr_val:<10.3f} {f1_val:<10.3f}")

    print("=" * 70)

    return recall_fpr


def calculate_value_scores(thresholds, recall_fpr, clv=4400.30, discount=300):
    """
    Calculate business value scores for different thresholds.

    The value score formula: (Recall * Net Value per Save) - (FPR * Discount Cost)

    Parameters
    ----------
    thresholds : list
        Threshold values to evaluate
    recall_fpr : dict
        Dictionary with 'recall' and 'fpr' lists from threshold analysis
    clv : float
        Customer Lifetime Value (default: $4,400.30)
    discount : float
        Retention discount cost (default: $300)

    Returns
    -------
    tuple
        (value_scores list, optimal_threshold, optimal_index)
    """
    net_value_per_save = clv - discount

    value_scores = []
    for recall, fpr in zip(recall_fpr['recall'], recall_fpr['fpr']):
        value_score = (recall * net_value_per_save) - (fpr * discount)
        value_scores.append(value_score)

    print("=" * 80)
    print(f"Value Score Analysis (CLV: ${clv:,.2f}, Discount: ${discount})")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Recall':<10} {'FPR':<10} {'Value Score':<12}")
    print("-" * 80)

    for th, recall, fpr, score in zip(thresholds, recall_fpr['recall'], recall_fpr['fpr'], value_scores):
        print(f"{th:<12} {recall:<10.3f} {fpr:<10.3f} ${score:<11,.2f}")

    print("=" * 80)

    # Find optimal threshold
    optimal_idx = np.argmax(value_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = value_scores[optimal_idx]

    print(f"\nOPTIMAL THRESHOLD: {optimal_threshold}")
    print(f"  Value Score: ${optimal_score:,.2f}")
    print(f"  Recall: {recall_fpr['recall'][optimal_idx]:.1%}")
    print(f"  FPR: {recall_fpr['fpr'][optimal_idx]:.1%}")

    return value_scores, optimal_threshold, optimal_idx
