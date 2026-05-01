import pandas as pd
from .greedy import GreedyOptimizer


class AmbulanceOptimizer:
    def __init__(self, df_traveltimes: pd.DataFrame, df_demand: pd.DataFrame, max_response_time: int = 900):
        self.df_traveltimes = df_traveltimes.copy()
        self.df_traveltimes.index = self.df_traveltimes.index.map(str)
        self.df_traveltimes.columns = self.df_traveltimes.columns.map(str)

        self.df_demand = df_demand.copy()
        self.df_demand["Postal codes"] = self.df_demand["Postal codes"].astype(str).str.strip()
        self.df_demand["Demand"] = pd.to_numeric(self.df_demand["Demand"], errors="coerce").fillna(0)

        self.max_response_time = max_response_time

    def _demand_lookup(self) -> pd.Series:
        return (
            self.df_demand.drop_duplicates(subset=["Postal codes"])
            .set_index("Postal codes")["Demand"]
        )

    def strategy_max_coverage(self, num_bases: int) -> dict:
        # Let op: dit is greedy max coverage, niet exact optimum
        optimizer = GreedyOptimizer(max_locations=num_bases, max_response_time=self.max_response_time)
        result = optimizer.find_optimal_locations(self.df_traveltimes, self.df_demand)
        result["strategy"] = "max_coverage_greedy"
        return result

    def strategy_max_speed(self, num_bases: int) -> dict:
        demand_lookup = self._demand_lookup()
        uncovered = set(self.df_traveltimes.columns)
        chosen_bases, covered, round_details = [], set(), []

        for ronde in range(1, num_bases + 1):
            best_base, best_avg_time, best_people, best_reachable = None, float("inf"), -1.0, []

            for base in self.df_traveltimes.index:
                row = self.df_traveltimes.loc[base]
                reachable = [loc for loc in uncovered if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time]
                if not reachable:
                    continue

                avg_time = float(pd.to_numeric(row[reachable], errors="coerce").mean())
                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())

                if (avg_time < best_avg_time) or (avg_time == best_avg_time and new_people > best_people):
                    best_base, best_avg_time, best_people, best_reachable = base, avg_time, new_people, reachable

            if best_base is None:
                break

            chosen_bases.append(best_base)
            newly_covered = set(best_reachable)
            uncovered -= newly_covered
            covered |= newly_covered

            round_details.append({
                "ronde": ronde,
                "basis_postcode": best_base,
                "gemiddelde_responstijd": best_avg_time,
                "nieuwe_mensen": best_people,
                "cumulatief_mensen": float(demand_lookup.reindex(list(covered)).fillna(0).sum()),
            })

        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

        return {
            "strategy": "max_speed",
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": len(covered),
            "total_locations": len(self.df_traveltimes.columns),
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0.0,
        }

    def strategy_balanced(self, num_bases: int, w_people: float = 0.8, w_speed: float = 0.2) -> dict:
        # Normalizeer gewichten veilig
        total_w = w_people + w_speed
        if total_w <= 0:
            raise ValueError("w_people + w_speed moet > 0 zijn.")
        w_people = w_people / total_w
        w_speed = w_speed / total_w

        demand_lookup = self._demand_lookup()
        uncovered = set(self.df_traveltimes.columns)
        chosen_bases, covered, round_details = [], set(), []

        for ronde in range(1, num_bases + 1):
            available_demand = float(demand_lookup.reindex(list(uncovered)).fillna(0).sum())
            if available_demand <= 0:
                break

            best_base, best_score, best_people, best_reachable, best_avg_time = None, -1.0, -1.0, [], 0.0

            for base in self.df_traveltimes.index:
                row = self.df_traveltimes.loc[base]
                reachable = [loc for loc in uncovered if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time]
                if not reachable:
                    continue

                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())
                avg_time = float(pd.to_numeric(row[reachable], errors="coerce").mean())

                people_score = new_people / available_demand
                speed_score = 1.0 - min(avg_time / self.max_response_time, 1.0)
                score = (w_people * people_score) + (w_speed * speed_score)

                if score > best_score:
                    best_base, best_score = base, score
                    best_people, best_reachable, best_avg_time = new_people, reachable, avg_time

            if best_base is None:
                break

            chosen_bases.append(best_base)
            newly_covered = set(best_reachable)
            uncovered -= newly_covered
            covered |= newly_covered

            round_details.append({
                "ronde": ronde,
                "basis_postcode": best_base,
                "nieuwe_mensen": best_people,
                "gemiddelde_responstijd": best_avg_time,
                "combined_score": best_score,
                "cumulatief_mensen": float(demand_lookup.reindex(list(covered)).fillna(0).sum()),
            })

        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

        return {
            "strategy": "balanced",
            "w_people": w_people,
            "w_speed": w_speed,
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": len(covered),
            "total_locations": len(self.df_traveltimes.columns),
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0.0,
        }

    def compare_strategies(self, num_bases: int = 5):
        cov = self.strategy_max_coverage(num_bases)
        spd = self.strategy_max_speed(num_bases)
        bal = self.strategy_balanced(num_bases)

        df = pd.DataFrame([
            {"Strategie": "Max Coverage (Greedy)", "Bases": ", ".join(cov["chosen_bases"]), "Bereikt": int(cov["covered_people"]), "Percentage": f"{cov['covered_people_pct']:.2f}%", "Postcodes": cov["covered_locations"]},
            {"Strategie": "Max Speed", "Bases": ", ".join(spd["chosen_bases"]), "Bereikt": int(spd["covered_people"]), "Percentage": f"{spd['covered_people_pct']:.2f}%", "Postcodes": spd["covered_locations"]},
            {"Strategie": "Balanced", "Bases": ", ".join(bal["chosen_bases"]), "Bereikt": int(bal["covered_people"]), "Percentage": f"{bal['covered_people_pct']:.2f}%", "Postcodes": bal["covered_locations"]},
        ])
        return df, {"max_coverage": cov, "max_speed": spd, "balanced": bal}

    def scan_balanced_weights(self, num_bases: int = 5, step: float = 0.05):
        """
        Test alle verhoudingen:
        w_people = 0.00..1.00 (stap = step), w_speed = 1 - w_people
        """
        if step <= 0 or step > 1:
            raise ValueError("step moet in (0, 1] liggen.")

        rows = []
        results = {}

        k = int(round(1.0 / step))
        for i in range(k + 1):
            w_people = round(i * step, 10)
            w_speed = round(1.0 - w_people, 10)

            res = self.strategy_balanced(num_bases=num_bases, w_people=w_people, w_speed=w_speed)
            key = f"{w_people:.2f}_{w_speed:.2f}"
            results[key] = res

            mean_rt = (
                pd.DataFrame(res["round_details"])["gemiddelde_responstijd"].mean()
                if res["round_details"] else float("nan")
            )

            rows.append({
                "w_people": round(w_people, 2),
                "w_speed": round(w_speed, 2),
                "bereikte_mensen": int(res["covered_people"]),
                "bereik_pct": round(res["covered_people_pct"], 2),
                "gedekte_postcodes": int(res["covered_locations"]),
                "gem_resp_tijd_sec": round(float(mean_rt), 2) if pd.notna(mean_rt) else None,
                "bases": ", ".join(res["chosen_bases"]),
            })

        df = pd.DataFrame(rows).sort_values(
            by=["bereikte_mensen", "gedekte_postcodes", "gem_resp_tijd_sec"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        return df, results