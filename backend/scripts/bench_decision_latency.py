# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, os, time, random, pathlib

# Add project root to Python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.smc.smc_fusion_core_enhanced import SMCFusionCoreEnhanced


def main():
    fusion = SMCFusionCoreEnhanced()
    N = 2000
    total = 0.0
    for i in range(N):
        state = random.choice(["T", "R", "B"])
        vol = random.choice(["Low", "Med", "High"])
        sess = random.choice(["ASIA", "EU", "US"])
        p_models = {
            "LSTM": random.random(),
            "PPO": random.random(),
            "XGB": random.random(),
            "CNN": random.random(),
        }
        t0 = time.perf_counter()
        d = fusion.decide(
            symbol="EURUSD",
            state=state,
            vol_bucket=vol,
            session=sess,
            spread_z=0.1,
            model_probs=p_models,
            meta={"spread_pips": 0.8},
        )
        total += (time.perf_counter() - t0) * 1000.0
    print(f"avg_ms={total/N:.3f}")


if __name__ == "__main__":
    main()
