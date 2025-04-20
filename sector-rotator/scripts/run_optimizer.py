# scripts/run_optimizer.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from optimizer.rule_based import generate_allocations

# Get raw weights from signal flags
weights = generate_allocations()  # DataFrame of shape (T, sectors), values in {-1, 0, +1}

# Apply weight cap if desired (e.g., max 50% in any one sector)
weights = weights.clip(upper=0.5, lower=-0.5)

# Normalize: sum of absolute weights = 1 (gross leverage control)
abs_sum = weights.abs().sum(axis=1).replace(0, 1)  # avoid divide-by-zero
weights = weights.div(abs_sum, axis=0)

# Save to file
weights.to_csv("data/weights/allocations.csv", index=False)

print("Saved → data/weights/allocations.csv")
