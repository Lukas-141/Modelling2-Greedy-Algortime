from pathlib import Path

import pandas as pd

from algorithms.optimizer import AmbulanceOptimizer

RUN_MODE = "sweep"
NUM_BASES = 5
MAX_RESPONSE_TIME = 900
EXPORT_CSV = False
SWEEP_STEP = 0.05


def load_data(src_dir: Path):
    excel = next(src_dir.glob("*.xlsx"), None) or next(src_dir.glob("*.xlsm"), None) or next(src_dir.glob("*.xls"), None)
    if excel is None:
        raise FileNotFoundError("Geen Excel-bestand gevonden in src/")

    raw = pd.read_excel(excel, sheet_name="Traveltimes (seconds)", header=None)
    reistijden = pd.DataFrame(
        raw.iloc[2:, 1:].values,
        index=[str(v) for v in raw.iloc[2:, 0].values],
        columns=[str(int(v)) if pd.notna(v) else f"col_{i}" for i, v in enumerate(raw.iloc[1, 1:].values)],
    ).apply(pd.to_numeric, errors="coerce").fillna(0)

    vraag = pd.read_excel(excel, sheet_name="Demand")
    vraag["Postal codes"] = vraag["Postal codes"].astype(str).str.strip()
    vraag["Demand"] = pd.to_numeric(vraag["Demand"], errors="coerce").fillna(0)
    return excel, reistijden, vraag


def fmt_time(value):
    return "-" if value is None or pd.isna(value) else f"{float(value):.1f} sec"


def avg_time(result: dict):
    if result.get("avg_response_time") is not None:
        return result["avg_response_time"]
    values = [r.get(key) for r in result.get("round_details", []) for key in ("gemiddelde_reistijd", "gemiddelde_responstijd", "gem_resp_tijd_sec") if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def summary_row(name: str, result: dict) -> dict:
    return {
        "Strategie": name,
        "Bases": ", ".join(result.get("chosen_bases", [])),
        "Bereikt": int(result.get("covered_people", 0)),
        "Percentage": f"{result.get('covered_people_pct', 0):.2f}%",
        "Postcodes": int(result.get("covered_locations", 0)),
        "Gem. reistijd": fmt_time(avg_time(result)),
    }


def run_coverage(opt: AmbulanceOptimizer):
    result = opt.strategy_max_coverage(NUM_BASES)
    table = pd.DataFrame([summary_row("Max Coverage", result)])
    print("\n=== MAX COVERAGE ===")
    print(table.to_string(index=False))
    return result, table


def run_compare(opt: AmbulanceOptimizer):
    _, details = opt.compare_strategies(num_bases=NUM_BASES)
    table = pd.DataFrame([
        summary_row("Max Coverage", details["max_coverage"]),
        summary_row("Max Speed", details["max_speed"]),
        summary_row("Balanced", details["balanced"]),
    ])
    print("\n=== VERGELIJKING ===")
    print(table.to_string(index=False))
    return details, table


def run_sweep(opt: AmbulanceOptimizer):
    weights, results = opt.scan_balanced_weights(num_bases=NUM_BASES, step=SWEEP_STEP)
    weights = weights.rename(columns={
        "gem_resp_tijd_sec": "Gem. reistijd",
        "bereikte_mensen": "Bereikt",
        "bereik_pct": "Percentage",
        "gedekte_postcodes": "Postcodes",
    })

    if {"w_people", "w_speed"}.issubset(weights.columns):
        weights["Verhouding"] = weights.apply(
            lambda row: f"{int(round(row['w_speed'] * 100)):>3}% speed / {int(round(row['w_people'] * 100)):>3}% bereik",
            axis=1,
        )
        weights = weights.drop(columns=["w_people", "w_speed"])

    print("\n=== BALANCED WEIGHT SWEEP ===")
    print(weights.to_string(index=False))
    return results, weights


def main():
    src_dir = Path(__file__).resolve().parent
    _, reistijden, vraag = load_data(src_dir)
    print(f"Reistijden: {reistijden.shape[0]} x {reistijden.shape[1]}")
    print(f"Totale vraag: {vraag['Demand'].sum():.0f}")

    optimizer = AmbulanceOptimizer(reistijden, vraag, max_response_time=MAX_RESPONSE_TIME)
    runners = {"coverage": run_coverage, "compare": run_compare, "sweep": run_sweep}
    if RUN_MODE not in runners:
        raise ValueError("RUN_MODE moet 'coverage', 'compare' of 'sweep' zijn.")

    _, output = runners[RUN_MODE](optimizer)
    if EXPORT_CSV and output is not None:
        out_dir = src_dir / "output"
        out_dir.mkdir(exist_ok=True)
        filenames = {
            "coverage": "coverage_summary.csv",
            "compare": "strategy_comparison.csv",
            "sweep": "balanced_weight_sweep.csv",
        }
        output.to_csv(out_dir / filenames[RUN_MODE], index=False)
        print(f"CSV opgeslagen: {out_dir / filenames[RUN_MODE]}")


if __name__ == "__main__":
    main()
