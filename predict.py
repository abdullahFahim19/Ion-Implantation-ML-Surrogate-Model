"""
predict.py
==========
Query the trained Gatekeeper-Constrained Dual-Stage RF Surrogate.

Loads the three pre-trained model files produced by train_model.py and
returns all nine ion implantation output quantities for a given set of
input parameters.

Usage (interactive):
    python predict.py

Usage (single prediction via CLI):
    python predict.py --substrate SiC --ion B --energy 300 --angle 7 --thickness 5000

Usage (batch CSV):
    python predict.py --batch my_conditions.csv --out predictions.csv
"""

import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONFIGURATION (must match train_model.py)
# ──────────────────────────────────────────────────────────────
INPUT_FEATURES   = ['substrate', 'ion', 'energy_keV', 'angle_deg', 'thickness_A']
TARGETS_GEOMETRY = ['Rp_A', 'dRp_A', 'lateral_range_A', 'lateral_straggle_A',
                    'radial_range_A', 'radial_straggle_A']
TARGETS_DAMAGE   = ['vacancies_per_ion', 'backscattered', 'transmitted']

VALID_SUBSTRATES = ['Si', 'SiC', 'GaAs']
VALID_IONS       = ['B', 'Mg', 'P', 'Ar', 'As']
MODEL_DIR        = 'models/'


# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
def load_models(model_dir: str = MODEL_DIR) -> tuple:
    """Load the three saved model pipelines."""
    paths = {
        'clf':     os.path.join(model_dir, 'gatekeeper_classifier.pkl'),
        'reg_geo': os.path.join(model_dir, 'geometry_regressor.pkl'),
        'reg_dmg': os.path.join(model_dir, 'damage_regressor.pkl'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"[ERROR] Model file not found: {path}")
            print("  Run  python train_model.py  first to generate model files.")
            sys.exit(1)

    clf     = joblib.load(paths['clf'])
    reg_geo = joblib.load(paths['reg_geo'])
    reg_dmg = joblib.load(paths['reg_dmg'])
    return clf, reg_geo, reg_dmg


# ──────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────
def predict(clf, reg_geo, reg_dmg, conditions: pd.DataFrame) -> pd.DataFrame:
    """
    Predict outputs for one or more implantation conditions.

    Parameters
    ----------
    conditions : DataFrame with columns matching INPUT_FEATURES.

    Returns
    -------
    DataFrame with columns:
        status, Rp_A, dRp_A, lateral_range_A, lateral_straggle_A,
        radial_range_A, radial_straggle_A, vacancies_per_ion,
        backscattered, transmitted
    """
    X = conditions[INPUT_FEATURES]

    # Stage 1 — Gatekeeper classification
    classes = clf.predict(X)

    results = []
    for idx, (_, row) in enumerate(conditions.iterrows()):
        query = pd.DataFrame([row[INPUT_FEATURES]])

        if classes[idx] == 0:
            # Transmission event — return physical defaults
            rec = {
                'status':             'TRANSMITTED',
                'Rp_A':               0.0,
                'dRp_A':              0.0,
                'lateral_range_A':    0.0,
                'lateral_straggle_A': 0.0,
                'radial_range_A':     0.0,
                'radial_straggle_A':  0.0,
                'vacancies_per_ion':  0.0,
                'backscattered':      0.0,
                'transmitted':        10000.0,
            }
        else:
            # Stopping event — run both regressors
            pred_geo = reg_geo.predict(query)[0]
            pred_dmg = reg_dmg.predict(query)[0]

            rec = {'status': 'STOPPED'}
            for col, val in zip(TARGETS_GEOMETRY, pred_geo):
                rec[col] = round(float(val), 2)
            for col, val in zip(TARGETS_DAMAGE, pred_dmg):
                rec[col] = round(float(val), 2)

        results.append(rec)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# DISPLAY HELPER
# ──────────────────────────────────────────────────────────────
def display_result(cond: dict, result: dict) -> None:
    print("\n" + "=" * 52)
    print(f"  Input : {cond['substrate']} | {cond['ion']} | "
          f"{cond['energy_keV']} keV | {cond['angle_deg']}° | "
          f"{cond['thickness_A']} Å")
    print(f"  Status: {result['status']}")
    print("-" * 52)

    if result['status'] == 'STOPPED':
        print("  DEPTH PROFILE")
        print(f"    Projected Range (Rp)      : {result['Rp_A']:.1f} Å")
        print(f"    Longitudinal Straggle (ΔRp): {result['dRp_A']:.1f} Å")
        print(f"    Lateral Range             : {result['lateral_range_A']:.1f} Å")
        print(f"    Lateral Straggle          : {result['lateral_straggle_A']:.1f} Å")
        print(f"    Radial Range              : {result['radial_range_A']:.1f} Å")
        print(f"    Radial Straggle           : {result['radial_straggle_A']:.1f} Å")
        print("  RADIATION DAMAGE")
        print(f"    Vacancies per Ion         : {result['vacancies_per_ion']:.1f}")
        print(f"    Backscattered Ions        : {result['backscattered']:.0f}")
        print(f"    Transmitted Ions          : {result['transmitted']:.0f}")
    else:
        print("  Ion energy exceeds substrate stopping capacity.")
        print("  -> Transmitted Ions : 10,000 (100%)")
        print("  -> Projected Range  : 0.0 Å")

    print("=" * 52)


# ──────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ──────────────────────────────────────────────────────────────
def interactive_mode(clf, reg_geo, reg_dmg) -> None:
    print("\n  Ion Implantation Surrogate — Interactive Query")
    print("  Type Ctrl+C to exit.\n")

    while True:
        try:
            print("Enter implantation parameters:")
            sub = input(f"  Substrate {VALID_SUBSTRATES}: ").strip()
            if sub not in VALID_SUBSTRATES:
                print("  Invalid substrate."); continue

            ion = input(f"  Ion       {VALID_IONS}: ").strip()
            if ion not in VALID_IONS:
                print("  Invalid ion."); continue

            energy    = float(input("  Energy    (keV) [10–10000]: "))
            angle     = float(input("  Tilt angle(deg) [0–89.9]  : "))
            thickness = float(input("  Thickness (Å)   [100–10000]: "))

            cond = {
                'substrate': sub, 'ion': ion,
                'energy_keV': energy, 'angle_deg': angle,
                'thickness_A': thickness,
            }
            df_q  = pd.DataFrame([cond])
            res_df = predict(clf, reg_geo, reg_dmg, df_q)
            display_result(cond, res_df.iloc[0].to_dict())

        except ValueError:
            print("  Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Query the trained ion implantation surrogate model.')
    parser.add_argument('--substrate', choices=VALID_SUBSTRATES,
                        help='Substrate material')
    parser.add_argument('--ion', choices=VALID_IONS,
                        help='Ion species')
    parser.add_argument('--energy', type=float,
                        help='Implantation energy (keV)')
    parser.add_argument('--angle', type=float, default=7.0,
                        help='Tilt angle in degrees (default: 7.0)')
    parser.add_argument('--thickness', type=float, default=5000.0,
                        help='Substrate thickness in Å (default: 5000)')
    parser.add_argument('--batch',
                        help='CSV file with multiple conditions to predict')
    parser.add_argument('--out',
                        help='Output CSV path for batch predictions')
    parser.add_argument('--model-dir', default=MODEL_DIR,
                        help=f'Directory with model files (default: {MODEL_DIR})')
    args = parser.parse_args()

    clf, reg_geo, reg_dmg = load_models(args.model_dir)
    print("Models loaded successfully.")

    # Batch mode
    if args.batch:
        df_batch = pd.read_csv(args.batch)
        res_df   = predict(clf, reg_geo, reg_dmg, df_batch)
        out_path = args.out or 'predictions.csv'
        pd.concat([df_batch.reset_index(drop=True),
                   res_df.reset_index(drop=True)], axis=1).to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")
        return

    # Single prediction
    if args.substrate and args.ion and args.energy:
        cond = {
            'substrate': args.substrate, 'ion': args.ion,
            'energy_keV': args.energy, 'angle_deg': args.angle,
            'thickness_A': args.thickness,
        }
        df_q   = pd.DataFrame([cond])
        res_df = predict(clf, reg_geo, reg_dmg, df_q)
        display_result(cond, res_df.iloc[0].to_dict())
        return

    # Default: interactive mode
    interactive_mode(clf, reg_geo, reg_dmg)


if __name__ == '__main__':
    main()
