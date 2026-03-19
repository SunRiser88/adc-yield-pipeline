import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

DEFECT_TYPES = ["Scratch", "Particle", "Pit", "Bridge", "Void", "Residue"]
DEFECT_COLORS = {
    "Scratch": "#ef4444",
    "Particle": "#f97316",
    "Pit": "#eab308",
    "Bridge": "#8b5cf6",
    "Void": "#06b6d4",
    "Residue": "#22c55e",
}
PROCESS_STEPS = ["Litho", "Etch", "CMP", "Deposition", "Implant", "Clean"]
LOTS = [f"LOT{str(i).zfill(4)}" for i in range(1, 21)]
WAFER_RADIUS = 150  # mm

def generate_wafer_defects(n_defects=None, defect_bias=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = n_defects or np.random.randint(20, 120)
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n)) * WAFER_RADIUS

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    if defect_bias:
        weights = [defect_bias.get(d, 1) for d in DEFECT_TYPES]
    else:
        weights = [3, 4, 2, 1, 2, 1]

    total = sum(weights)
    probs = [w / total for w in weights]
    defect_labels = np.random.choice(DEFECT_TYPES, size=n, p=probs)

    sizes = np.random.lognormal(mean=1.5, sigma=0.6, size=n)
    sizes = np.clip(sizes, 0.1, 20)

    confidence = np.random.beta(8, 2, size=n)

    return pd.DataFrame({
        "x": x,
        "y": y,
        "defect_type": defect_labels,
        "size_um": np.round(sizes, 2),
        "confidence": np.round(confidence, 3),
    })


def generate_lot_history(n_lots=20, n_wafers_per_lot=25):
    records = []
    base_date = datetime(2024, 1, 1)

    for i, lot in enumerate(LOTS[:n_lots]):
        lot_date = base_date + timedelta(days=i * 3)
        step = random.choice(PROCESS_STEPS)
        defect_bias_shift = random.choice([None, {"Scratch": 6}, {"Particle": 6}, {"Void": 5}])

        for wafer_id in range(1, n_wafers_per_lot + 1):
            df = generate_wafer_defects(defect_bias=defect_bias_shift)
            total = len(df)
            density = round(total / (np.pi * (WAFER_RADIUS / 10) ** 2), 4)
            yield_val = max(0, min(1, 1 - (density * np.random.uniform(0.3, 0.7))))

            counts = df["defect_type"].value_counts().to_dict()
            for dt in DEFECT_TYPES:
                counts.setdefault(dt, 0)

            records.append({
                "lot_id": lot,
                "wafer_id": wafer_id,
                "process_step": step,
                "date": lot_date,
                "total_defects": total,
                "defect_density": density,
                "yield_pct": round(yield_val * 100, 2),
                **{f"count_{dt.lower()}": counts[dt] for dt in DEFECT_TYPES},
            })

    return pd.DataFrame(records)


def generate_classification_results(n_samples=200):
    """Simulate ML classifier output with confusion matrix data."""
    true_labels = np.random.choice(DEFECT_TYPES, size=n_samples,
                                   p=[0.25, 0.30, 0.15, 0.10, 0.12, 0.08])
    noise_mask = np.random.random(n_samples) < 0.12
    pred_labels = true_labels.copy()
    for i in np.where(noise_mask)[0]:
        choices = [d for d in DEFECT_TYPES if d != true_labels[i]]
        pred_labels[i] = random.choice(choices)

    confidence = np.where(pred_labels == true_labels,
                          np.random.beta(9, 2, n_samples),
                          np.random.beta(4, 4, n_samples))

    return pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "confidence": np.round(confidence, 3),
        "correct": pred_labels == true_labels,
    })
