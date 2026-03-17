#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(
    description="Predict inhibitor and pIC50 from SMILES"
)
parser.add_argument("input_csv")
parser.add_argument("output_csv")
args = parser.parse_args()


# ============================================================
# Load input
# ============================================================
try:
    df = pd.read_csv(args.input_csv)
except Exception:
    sys.exit("ERROR: Cannot read input CSV")

if not {"Name", "Smiles"}.issubset(df.columns):
    sys.exit("ERROR: CSV must contain Name and Smiles")

if len(df) == 0:
    sys.exit("ERROR: Empty input file")

df = df.reset_index(drop=True)
df["_ID"] = [f"Mol_{i}" for i in range(len(df))]


# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parent

PADEL = BASE / "PaDEL" / "PaDEL-Descriptor.jar"
DESTYPE = BASE / "padel_destype.xml"
XTRAIN = BASE / "X_train.csv"
SCALER = BASE / "padel_scaler.pkl"
MODEL = BASE / "gsk3bmt_model.h5"

for f in [PADEL, DESTYPE, XTRAIN, SCALER, MODEL]:
    if not f.exists():
        sys.exit(f"ERROR: Missing file {f}")


# ============================================================
# Temp workspace
# ============================================================
tmp = Path(tempfile.mkdtemp())
padel_dir = tmp / "padel"
padel_dir.mkdir()

try:
    # ========================================================
    # Write SMILES (internal IDs)
    # ========================================================
    smi = padel_dir / "mol.smi"
    with open(smi, "w") as f:
        for _, r in df.iterrows():
            f.write(f"{r['Smiles']} {r['_ID']}\n")

    # ========================================================
    # Run PaDEL
    # ========================================================
    desc_file = tmp / "desc.csv"

    config = tmp / "padel.cfg"
    with open(config, "w") as f:
        f.write(
    f"""Compute2D=true
    Compute3D=false
    ComputeFingerprints=true
    Convert3D=No
    Directory={padel_dir}
    DescriptorFile={desc_file}
    DetectAromaticity=true
    Log=true
    MaxCpdPerFile=0
    MaxJobsWaiting=-1
    MaxRunTime=600000
    MaxThreads=-1
    RemoveSalt=true
    Retain3D=false
    RetainOrder=false
    StandardizeNitro=true
    StandardizeTautomers=false
    TautomerFile=
    UseFilenameAsMolName=false
    """
        )

    subprocess.run(
        ["java", "-jar", str(PADEL), "-config", str(config), "-descriptortypes", str(DESTYPE)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not desc_file.exists() or desc_file.stat().st_size == 0:
        sys.exit("ERROR: PaDEL failed")

    desc = pd.read_csv(desc_file)

    # ========================================================
    # Merge (using INTERNAL ID)
    # ========================================================
    merged = pd.merge(df, desc, left_on="_ID", right_on="Name", how="left")

    failed = merged.isna().any(axis=1)
    valid = ~failed

    if valid.sum() == 0:
        # all failed
        out = pd.DataFrame({
            "Name": df["Name"],
            "Predicted_Prob": np.nan,
            "Predicted_Class": "Failed",
            "Predicted_pIC50": np.nan,
            "Predicted_IC50_uM": np.nan,
            "Predicted_IC50_M": np.nan,
        })
        out.to_csv(args.output_csv, index=False)
        sys.exit(0)

    # ========================================================
    # Features
    # ========================================================
    X_train = pd.read_csv(XTRAIN)
    feats = [c for c in X_train.columns if c != "Name"]
    means = X_train[feats].mean()

    X = merged.loc[valid].drop(columns=["Name_x", "Name_y", "_ID"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    for c in feats:
        if c not in X:
            X[c] = means[c]

    X = X.reindex(columns=feats)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(means)

    # ========================================================
    # Scale + predict
    # ========================================================
    scaler = joblib.load(SCALER)
    model = load_model(MODEL, compile=False)

    Xs = scaler.transform(X)
    pred = model.predict(Xs, verbose=0)

    prob = pred[0].ravel()
    pic50 = pred[1].ravel()

    # ========================================================
    # Build output
    # ========================================================
    n = len(df)

    prob_full = np.full(n, np.nan)
    pic50_full = np.full(n, np.nan)

    prob_full[valid.values] = prob
    pic50_full[valid.values] = pic50

    label = np.where(prob_full >= 0.5, "Inhibitor", "Non-inhibitor")
    label[~valid.values] = "Failed"

    ic50_uM = 10 ** (6 - pic50_full)
    ic50_M = 10 ** (-pic50_full)

    ic50_uM[~valid.values] = np.nan
    ic50_M[~valid.values] = np.nan

    out = pd.DataFrame({
        "Name": df["Name"],
        "Predicted_Prob": prob_full,
        "Predicted_Class": label,
        "Predicted_pIC50": pic50_full,
        "Predicted_IC50_uM": ic50_uM,
        "Predicted_IC50_M": ic50_M,
    })

    out.to_csv(args.output_csv, index=False)
    print(f"Saved → {args.output_csv}")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
