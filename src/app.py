from pathlib import Path
import pandas as pd
from algorithms.optimizer import AmbulanceOptimizer

# ===== KIES 1 MODUS =====
# "coverage" -> alleen Max Coverage
# "compare"  -> 3 strategieën vergelijken
# "sweep"    -> alleen balanced weight sweep
RUN_MODE = "compare"

NUM_BASES = 5
MAX_RESPONSE_TIME = 900
EXPORT_CSV = False
SWEEP_STEP = 0.05


def print_section(title: str, width: int = 90):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def load_data_from_excel(src_dir: Path):
    excel_files = list(src_dir.glob("*.xlsx")) + list(src_dir.glob("*.xlsm")) + list(src_dir.glob("*.xls"))
    if not excel_files:
        raise FileNotFoundError("Geen Excel-bestand gevonden in src/")

    excel_path = excel_files[0]
    print(f"Gebruik Excel-bestand: {excel_path.name}")

    # Traveltimes in jullie layout
    df_tt_raw = pd.read_excel(excel_path, sheet_name="Traveltimes (seconds)", header=None)
    col_names = [str(int(x)) if pd.notna(x) else f"col_{i}" for i, x in enumerate(df_tt_raw.iloc[1, 1:].values)]
    row_names = [str(int(x)) if pd.notna(x) else f"row_{i}" for i, x in enumerate(df_tt_raw.iloc[2:, 0].values)]
    data = df_tt_raw.iloc[2:, 1:].values

    df_traveltimes = pd.DataFrame(data, index=row_names, columns=col_names)
    df_traveltimes = df_traveltimes.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Demand
    df_demand = pd.read_excel(excel_path, sheet_name="Demand")
    df_demand["Postal codes"] = df_demand["Postal codes"].astype(str).str.strip()
    df_demand["Demand"] = pd.to_numeric(df_demand["Demand"], errors="coerce").fillna(0)

    return excel_path, df_traveltimes, df_demand


def run_coverage(opt: AmbulanceOptimizer):
    res = opt.strategy_max_coverage(NUM_BASES)

    print_section("MAX COVERAGE")
    print("Gekozen basislocaties:")
    for i, b in enumerate(res["chosen_bases"], start=1):
        print(f"{i}. {b}")

    print("\nPer ronde:")
    for r in res["round_details"]:
        print(
            f"Ronde {r['ronde']}: basis {r['basis_postcode']} | "
            f"nieuwe mensen {r['nieuwe_mensen']:.0f} | "
            f"cumulatief {r['cumulatief_mensen']:.0f}"
        )

    print("\nSamenvatting:")
    print(f"Postcodes: {res['covered_locations']} / {res['total_locations']}")
    print(f"Mensen: {res['covered_people']:.0f} / {res['total_people']:.0f} ({res['covered_people_pct']:.2f}%)")
    return {"max_coverage": res}


def run_compare(opt: AmbulanceOptimizer):
    df_comparison, details = opt.compare_strategies(num_bases=NUM_BASES)
    print_section("VERGELIJKING STRATEGIEËN")
    print(df_comparison.to_string(index=False))
    return details, df_comparison


def run_sweep(opt: AmbulanceOptimizer):
    df_weights, weight_results = opt.scan_balanced_weights(num_bases=NUM_BASES, step=SWEEP_STEP)
    print_section("BALANCED WEIGHT SWEEP")
    print(df_weights.head(15).to_string(index=False))

    best = df_weights.iloc[0]
    print("\nBeste verhouding:")
    print(
        f"w_people={best['w_people']:.2f}, w_speed={best['w_speed']:.2f} | "
        f"bereikte_mensen={best['bereikte_mensen']} ({best['bereik_pct']:.2f}%)"
    )
    return weight_results, df_weights


def main():
    src_dir = Path(__file__).resolve().parent
    _, df_traveltimes, df_demand = load_data_from_excel(src_dir)

    print(f"Traveltimes: {df_traveltimes.shape[0]} x {df_traveltimes.shape[1]}")
    print(f"Totale demand: {df_demand['Demand'].sum():.0f}")

    opt = AmbulanceOptimizer(
        df_traveltimes=df_traveltimes,
        df_demand=df_demand,
        max_response_time=MAX_RESPONSE_TIME
    )

    details = {}
    df_out = None

    if RUN_MODE == "coverage":
        details = run_coverage(opt)
    elif RUN_MODE == "compare":
        details, df_out = run_compare(opt)
    elif RUN_MODE == "sweep":
        details, df_out = run_sweep(opt)
    else:
        raise ValueError("RUN_MODE moet 'coverage', 'compare' of 'sweep' zijn.")

    if EXPORT_CSV and df_out is not None:
        out_dir = src_dir / "output"
        out_dir.mkdir(exist_ok=True)
        if RUN_MODE == "compare":
            df_out.to_csv(out_dir / "strategy_comparison.csv", index=False)
        elif RUN_MODE == "sweep":
            df_out.to_csv(out_dir / "balanced_weight_sweep.csv", index=False)
        print(f"\nCSV export: {out_dir}")


if __name__ == "__main__":
    main()
