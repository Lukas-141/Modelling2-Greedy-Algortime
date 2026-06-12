import pandas as pd


class GreedyOptimizer:
    def __init__(self, max_locations: int = 5, max_response_time: int = 900):
        self.max_locations = max_locations
        self.max_response_time = max_response_time

    def set_max_locations(self, max_locations: int):
        self.max_locations = max_locations

    def _demand_lookup(self, df_demand: pd.DataFrame) -> pd.Series:
        demand = df_demand.copy()
        demand["Postal codes"] = demand["Postal codes"].astype(str).str.strip()
        demand["Demand"] = pd.to_numeric(demand["Demand"], errors="coerce").fillna(0)
        return demand.drop_duplicates(subset=["Postal codes"]).set_index("Postal codes")["Demand"]

    def find_optimal_locations(self, df_traveltimes: pd.DataFrame, df_demand: pd.DataFrame) -> dict:
        tt = df_traveltimes.copy()
        tt.index = tt.index.map(str)
        tt.columns = tt.columns.map(str)
        demand_lookup = self._demand_lookup(df_demand)

        uncovered = set(tt.columns)
        chosen_bases, covered, round_details = [], set(), []

        for ronde in range(1, self.max_locations + 1):
            best = None

            for base in tt.index:
                row = tt.loc[base]
                reachable = [loc for loc in uncovered if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time]
                if not reachable:
                    continue

                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())
                score = (new_people, len(reachable), -float(pd.to_numeric(row[reachable], errors="coerce").mean()))
                if best is None or score > best["score"]:
                    best = {"base": base, "reachable": reachable, "people": new_people, "score": score}

            if best is None:
                break

            chosen_bases.append(best["base"])
            newly_covered = set(best["reachable"])
            uncovered -= newly_covered
            covered |= newly_covered
            round_details.append({
                "ronde": ronde,
                "basis_postcode": best["base"],
                "nieuwe_postcodes": len(newly_covered),
                "nieuwe_mensen": best["people"],
                "cumulatief_mensen": float(demand_lookup.reindex(list(covered)).fillna(0).sum()),
            })

        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())
        return {
            "strategy": "max_coverage",
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": len(covered),
            "total_locations": len(tt.columns),
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0,
        }