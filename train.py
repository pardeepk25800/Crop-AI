# train.py — One-Click Training Script for CropAI
# Runs: Data Generation → Disease Model Training → Yield Model Training

import os
import time
import json
import argparse

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main(args):
    total_start = time.time()

    banner("CropAI — Full Training Pipeline")
    print(f"  Disease model samples : {args.samples_per_class} per class")
    print(f"  Yield dataset rows    : {args.yield_rows}")
    print(f"  Epochs (disease)      : {args.epochs}")

    # ── STEP 1: Generate Data ─────────────────────────────────────────────
    banner("Step 1 / 3 — Data Generation")
    from data_generator import generate_disease_dataset, generate_yield_dataset
    from config import DISEASE_DATA_DIR, YIELD_DATA_PATH

    if args.regenerate or not os.path.isdir(DISEASE_DATA_DIR):
        generate_disease_dataset(samples_per_class=args.samples_per_class)
    else:
        print(f"[Step1] Disease images found at {DISEASE_DATA_DIR} — skipping")

    if args.regenerate or not os.path.exists(YIELD_DATA_PATH):
        generate_yield_dataset(n_rows=args.yield_rows)
    else:
        print(f"[Step1] Yield CSV found at {YIELD_DATA_PATH} — skipping")

    # ── STEP 2: Train Disease Model ───────────────────────────────────────
    banner("Step 2 / 3 — CNN Disease Detection Training")
    from config import EPOCHS
    import config
    config.EPOCHS = args.epochs  # override if passed

    from disease_model import train_model as train_disease
    disease_model, classes, history = train_disease()

    # ── STEP 3: Train Yield Model ─────────────────────────────────────────
    banner("Step 3 / 3 — Yield Prediction Training")
    from yield_model import train_yield_model
    ensemble, metrics = train_yield_model()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    banner("✅ Training Complete")
    print(f"  Total time     : {elapsed/60:.1f} minutes")
    print(f"  Disease classes: {len(classes)}")
    print(f"  Best val acc   : {max(history['val_acc'])*100:.2f}%")
    print(f"  Yield R²       : {metrics['r2']:.4f}")
    print(f"  Yield MAE      : {metrics['mae']:.2f} kg/ha")
    print()
    print("  Next steps:")
    print("  >> Start API   : uvicorn api:app --reload")
    print("  >> Open UI     : streamlit run streamlit_app.py")
    print("  >> API Docs    : http://localhost:8000/docs")

    results = {
        "disease_accuracy": max(history["val_acc"]),
        "yield_r2":         metrics["r2"],
        "yield_mae":        metrics["mae"],
        "training_time_s":  elapsed,
    }
    with open("results/training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved → results/training_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropAI Training Pipeline")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Leaf images per disease class (default: 50)")
    parser.add_argument("--yield-rows", type=int, default=5000,
                        help="Rows in synthetic yield CSV (default: 5000)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs for CNN (default: 25)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate datasets even if they exist")
    args = parser.parse_args()
    main(args)
