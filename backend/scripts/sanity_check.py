import sys
import os
import pandas as pd

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.data_sanity import normalize_timestamps, normalize_volume

# simulate sample

df = pd.DataFrame(
    {
        "timestamp": [0, 300000, 600000],  # suspicious ms from epoch
        "open": [20.53, 20.54, 20.53],
        "high": [20.55, 20.56, 20.54],
        "low": [20.52, 20.53, 20.51],
        "close": [20.53, 20.54, 20.53],
        "volume": [1368209693670, 1133442532006, 1132777952951],
    }
)

print("Before normalization:\n", df.head())

df = normalize_timestamps(df)
df, scale = normalize_volume(df)

print("\nAfter normalization:\n", df.head())
print("Scale applied:", scale)
