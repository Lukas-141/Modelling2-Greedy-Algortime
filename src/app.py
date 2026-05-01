from pathlib import Path
import pandas as pd
from algorithms.optimizer import AmbulanceOptimizer

# ===== KIES 1 MODUS =====
# "coverage" -> alleen Max Coverage
# "compare"  -> 3 strategieën vergelijken
# "sweep"    -> alleen balanced weight sweep
RUN_MODE = "sweep"

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


def _fmt_time(value):
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.1f} sec"
    except Exception:
        return "-"


def _extract_avg_time(result: dict):
    """
    Probeert de gemiddelde reistijd te pakken uit bestaande output,
    zonder het algoritme aan te passen.
    """
    if result.get("avg_response_time") is not None:
        return result.get("avg_response_time")

    round_details = result.get("round_details", [])
    if not round_details:
        return None

    times = []
    for r in round_details:
        for key in ("gemiddelde_reistijd", "gemiddelde_responstijd", "gem_resp_tijd_sec"):
            if key in r and r[key] is not None and not pd.isna(r[key]):
                times.append(float(r[key]))
                break

    if not times:
        return None

    return sum(times) / len(times)


def _strategy_summary(name: str, res: dict) -> dict:
    return {
        "Strategie": name,
        "Bases": ", ".join(res.get("chosen_bases", [])),
        "Bereikt": int(res.get("covered_people", 0)),
        "Percentage": f"{res.get('covered_people_pct', 0):.2f}%",
        "Postcodes": int(res.get("covered_locations", 0)),
        "Gem. reistijd": _fmt_time(_extract_avg_time(res)),
    }


def _normalize_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "gem_resp_tijd_sec": "Gem. reistijd",
        "Gem. reistijd (sec)": "Gem. reistijd",
        "bereikte_mensen": "Bereikt",
        "bereik_pct": "Percentage",
        "gedekte_postcodes": "Postcodes",
    }
    out = df.copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def run_coverage(opt: AmbulanceOptimizer):
    res = opt.strategy_max_coverage(NUM_BASES)

    print_section("MAX COVERAGE")
    print("Gekozen basislocaties:")
    for i, b in enumerate(res["chosen_bases"], start=1):
        print(f"{i}. {b}")

    print("\nPer ronde:")
    for r in res["round_details"]:
        avg_time = r.get("gemiddelde_reistijd", r.get("gemiddelde_responstijd"))
        print(
            f"Ronde {r['ronde']}: basis {r['basis_postcode']} | "
            f"nieuwe mensen {r['nieuwe_mensen']:.0f} | "
            f"gem. reistijd {_fmt_time(avg_time)} | "
            f"cumulatief {r['cumulatief_mensen']:.0f}"
        )

    print("\nSamenvatting:")
    print(f"Postcodes: {res['covered_locations']} / {res['total_locations']}")
    print(f"Mensen: {res['covered_people']:.0f} / {res['total_people']:.0f} ({res['covered_people_pct']:.2f}%)")
    print(f"Gemiddelde reistijd: {_fmt_time(_extract_avg_time(res))}")

    df_summary = pd.DataFrame([_strategy_summary("Max Coverage", res)])
    print("\nKorte samenvatting:")
    print(df_summary.to_string(index=False))

    return {"max_coverage": res}, df_summary


def run_compare(opt: AmbulanceOptimizer):
    df_comparison, details = opt.compare_strategies(num_bases=NUM_BASES)

    print_section("VERGELIJKING STRATEGIEËN")
    # uniforme output, inclusief reistijd
    df_summary = pd.DataFrame([
        _strategy_summary("Max Coverage", details["max_coverage"]),
        _strategy_summary("Max Speed", details["max_speed"]),
        _strategy_summary("Balanced", details["balanced"]),
    ])
    print(df_summary.to_string(index=False))

    return details, df_summary


def run_sweep(opt: AmbulanceOptimizer):
    # 100% speed -> 100% bereik
    df_weights, weight_results = opt.scan_balanced_weights(num_bases=NUM_BASES, step=SWEEP_STEP)
    df_weights = _normalize_weight_columns(df_weights)

    # Zorg dat de kolommen netjes en voorspelbaar zijn
    if "w_people" in df_weights.columns and "w_speed" in df_weights.columns:
        df_weights = df_weights.sort_values(
            by=["w_people", "Bereikt", "Postcodes"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

    print_section("BALANCED WEIGHT SWEEP (100% speed → 100% bereik)")

    # Mooie tabel met duidelijke kolomnamen
    display_cols = []
    for col in ["w_speed", "w_people", "Bereikt", "Percentage", "Postcodes", "Gem. reistijd", "bases"]:
        if col in df_weights.columns:
            display_cols.append(col)

    table = df_weights[display_cols].copy()

    # Extra leesbare labels
    if "w_speed" in table.columns and "w_people" in table.columns:
        table.insert(0, "Verhouding", table.apply(
            lambda r: f"{int(round(r['w_speed'] * 100)):>3}% speed / {int(round(r['w_people'] * 100)):>3}% bereik",
            axis=1
        ))
        table = table.drop(columns=["w_speed", "w_people"])

    # Alleen hele tabel tonen
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print(table.to_string(index=False))

    best = df_weights.iloc[0]
    print("\nBeste verhouding:")
    print(
        f"Speed {best.get('w_speed', 0) * 100:.0f}% / Bereik {best.get('w_people', 0) * 100:.0f}% | "
        f"Bereikt={best.get('Bereikt', best.get('bereikte_mensen', '-'))} | "
        f"Percentage={best.get('Percentage', best.get('bereik_pct', '-'))} | "
        f"Postcodes={best.get('Postcodes', best.get('gedekte_postcodes', '-'))} | "
        f"Gem. reistijd={best.get('Gem. reistijd', '-')}"
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
        details, df_out = run_coverage(opt)
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
        elif RUN_MODE == "coverage":
            df_out.to_csv(out_dir / "coverage_summary.csv", index=False)
        print(f"\nCSV export: {out_dir}")


if __name__ == "__main__":
    main()
