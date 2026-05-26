"""
train_model.py
==============
Gatekeeper-Constrained Dual-Stage Random Forest Surrogate
for Ion Implantation Simulation.

Trains a two-stage ML pipeline that emulates SRIM/BCA physics:
  Stage 1 — Gatekeeper Random Forest Classifier
             (discriminates stopped vs transmitted ions)
  Stage 2 — Two Random Forest Regressors
             (geometry outputs + damage/vacancy output)

Reference:
    Fahim et al., "Emulating Binary Collision Approximation Physics
    via a Gatekeeper-Constrained Dual-Stage Random Forest",
    Computational Materials Science (2025).

Usage:
    python train_model.py
    python train_model.py --data sample_database.csv --out models/
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
INPUT_FEATURES   = ['substrate', 'ion', 'energy_keV', 'angle_deg', 'thickness_A']
CATEGORICAL_COLS = ['substrate', 'ion']
NUMERICAL_COLS   = ['energy_keV', 'angle_deg', 'thickness_A']

TARGET_CLASS     = 'is_profile'           # 1 = stopped, 0 = transmitted

TARGETS_GEOMETRY = [                       # 6 spatial outputs
    'Rp_A', 'dRp_A',
    'lateral_range_A', 'lateral_straggle_A',
    'radial_range_A',  'radial_straggle_A',
]
TARGETS_DAMAGE   = ['vacancies_per_ion',   # 3 interaction-physics outputs
                    'backscattered',
                    'transmitted']


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Standardise substrate labels produced by older SRIM batch scripts
    df['substrate'] = df['substrate'].replace({'Ga50As50': 'GaAs', 'Si50C50': 'SiC'})

    # Keep only clean binary class labels
    df = df[df[TARGET_CLASS].isin([0, 1])].copy()
    df = df.dropna(subset=INPUT_FEATURES)

    print(f"  Loaded {len(df)} simulations "
          f"({df[TARGET_CLASS].sum()} stopped, "
          f"{(df[TARGET_CLASS]==0).sum()} transmitted)")
    return df


# ──────────────────────────────────────────────────────────────
# PREPROCESSOR BUILDERS
# ──────────────────────────────────────────────────────────────
def build_preprocessor(scaler='standard') -> ColumnTransformer:
    """Return a ColumnTransformer with one-hot encoding + chosen scaler."""
    num_scaler = StandardScaler() if scaler == 'standard' else RobustScaler()
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
        ('num', num_scaler, NUMERICAL_COLS),
    ], remainder='drop')


# ──────────────────────────────────────────────────────────────
# STAGE 1 — GATEKEEPER CLASSIFIER
# ──────────────────────────────────────────────────────────────
def train_classifier(df: pd.DataFrame, cv_folds: int = 10) -> Pipeline:
    """Train and cross-validate the gatekeeper RF classifier."""
    print("\nStage 1 — Gatekeeper Classifier")

    X = df[INPUT_FEATURES]
    y = df[TARGET_CLASS]

    pipe = Pipeline([
        ('pre', build_preprocessor('standard')),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    f1_scores  = cross_val_score(pipe, X, y, cv=cv, scoring='f1')

    print(f"  {cv_folds}-fold CV  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"  {cv_folds}-fold CV  F1-score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Fit on full dataset
    pipe.fit(X, y)
    print("  Classifier trained on full dataset.")
    return pipe


# ──────────────────────────────────────────────────────────────
# STAGE 2 — REGRESSORS (stopping events only)
# ──────────────────────────────────────────────────────────────
def train_regressors(df: pd.DataFrame, cv_folds: int = 10) -> tuple:
    """
    Train two RF regressors on the stopped-ion subset:
      - Geometry regressor  (Rp, ΔRp, Rlat, ΔRlat, Rrad, ΔRrad)
      - Damage regressor    (Vvac, Nback, Ntrans)
    Returns (reg_geometry, reg_damage).
    """
    print("\nStage 2 — Regressors (stopped events only)")

    df_s = df[df[TARGET_CLASS] == 1].copy()
    df_s = df_s.dropna(subset=TARGETS_GEOMETRY + TARGETS_DAMAGE)
    X_s  = df_s[INPUT_FEATURES]

    print(f"  Training subset: {len(df_s)} stopped events")

    # ── Geometry regressor ────────────────────────────────
    y_geo = df_s[TARGETS_GEOMETRY]

    reg_geo = Pipeline([
        ('pre', build_preprocessor('standard')),
        ('reg', RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    cv_kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2_geo = cross_val_score(reg_geo, X_s, y_geo, cv=cv_kf, scoring='r2')
    print(f"  Geometry regressor  {cv_folds}-fold CV  R²: {r2_geo.mean():.3f} ± {r2_geo.std():.3f}")

    reg_geo.fit(X_s, y_geo)

    # ── Damage regressor ──────────────────────────────────
    y_dmg = df_s[TARGETS_DAMAGE]

    reg_dmg = Pipeline([
        ('pre', build_preprocessor('robust')),
        ('reg', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    r2_dmg = cross_val_score(reg_dmg, X_s, y_dmg, cv=cv_kf, scoring='r2')
    print(f"  Damage regressor    {cv_folds}-fold CV  R²: {r2_dmg.mean():.3f} ± {r2_dmg.std():.3f}")

    reg_dmg.fit(X_s, y_dmg)

    print("  Both regressors trained on full stopped-event subset.")
    return reg_geo, reg_dmg


# ──────────────────────────────────────────────────────────────
# SAVE MODELS
# ──────────────────────────────────────────────────────────────
def save_models(clf, reg_geo, reg_dmg, out_dir: str = 'models/') -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf,     os.path.join(out_dir, 'gatekeeper_classifier.pkl'))
    joblib.dump(reg_geo, os.path.join(out_dir, 'geometry_regressor.pkl'))
    joblib.dump(reg_dmg, os.path.join(out_dir, 'damage_regressor.pkl'))
    print(f"\nModels saved to '{out_dir}'")
    print("  gatekeeper_classifier.pkl")
    print("  geometry_regressor.pkl")
    print("  damage_regressor.pkl")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train ion implantation surrogate models.')
    parser.add_argument('--data', default='sample_database.csv',
                        help='Path to training CSV (default: sample_database.csv)')
    parser.add_argument('--out',  default='models/',
                        help='Directory to save trained models (default: models/)')
    parser.add_argument('--cv',   type=int, default=10,
                        help='Number of cross-validation folds (default: 10)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Gatekeeper-Constrained Dual-Stage RF Surrogate")
    print("  Ion Implantation Emulator — Training Script")
    print("=" * 60)

    df = load_data(args.data)
    clf         = train_classifier(df, cv_folds=args.cv)
    reg_geo, reg_dmg = train_regressors(df, cv_folds=args.cv)
    save_models(clf, reg_geo, reg_dmg, out_dir=args.out)

    print("\nTraining complete.")
    print("Run  python predict.py  to query the trained models.")


if __name__ == '__main__':
    main()
