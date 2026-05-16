from __future__ import annotations

from collections import Counter


def precision_recall_f1(rows: list[dict[str, str]], *, positive_label: str = 'supports') -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        predicted = str(row.get('predicted_label', ''))
        truth = str(row.get('truth_label', ''))
        if predicted == positive_label and truth == positive_label:
            tp += 1
        elif predicted == positive_label and truth != positive_label:
            fp += 1
        elif predicted != positive_label and truth == positive_label:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def website_accuracy(rows: list[dict[str, str]]) -> dict[str, float]:
    website_rows = [row for row in rows if row.get('attribute') == 'website']
    if not website_rows:
        return {'website_accuracy': 0.0, 'website_rows': 0}

    correct = sum(1 for row in website_rows if str(row.get('predicted_value', '')) == str(row.get('truth_value', '')))
    return {'website_accuracy': correct / len(website_rows), 'website_rows': len(website_rows)}


def label_distribution(rows: list[dict[str, str]], label_key: str = 'weak_label') -> dict[str, int]:
    return dict(Counter(str(row.get(label_key, '')) for row in rows))
