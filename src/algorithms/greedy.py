import pandas as pd


class GreedyOptimizer:
    def __init__(self, max_locations: int = 5, max_response_time: int = 900):
        self.max_locations = max_locations
        self.max_response_time = max_response_time

    def set_max_locations(self, max_locations: int):
        self.max_locations = max_locations

    def find_optimal_locations(self, df_traveltimes: pd.DataFrame, df_demand: pd.DataFrame) -> dict:
        demand_lookup = (
            df_demand.drop_duplicates(subset=["Postal codes"])
            .set_index("Postal codes")["Demand"]
        )

        uncovered = set(df_traveltimes.columns.astype(str))
        chosen_bases = []
        covered = set()
        round_details = []

        # Zorg voor string labels zodat lookup consistent is
        tt = df_traveltimes.copy()
        tt.index = tt.index.map(str)
        tt.columns = tt.columns.map(str)

        for ronde in range(1, self.max_locations + 1):
            best_base = None
            best_people = -1.0
            best_count = -1
            best_avg_time = float("inf")
            best_reachable = []

            for base in tt.index:
                row = tt.loc[base]
                reachable = [
                    loc for loc in uncovered
                    if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time
                ]
                if not reachable:
                    continue

                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())
                new_count = len(reachable)
                avg_time = float(pd.to_numeric(row[reachable], errors="coerce").mean())

                # BELANGRIJK: primair op mensen, niet op aantal postcodes
                if (
                    (new_people > best_people) or
                    (new_people == best_people and new_count > best_count) or
                    (new_people == best_people and new_count == best_count and avg_time < best_avg_time)
                ):
                    best_base = base
                    best_people = new_people
                    best_count = new_count
                    best_avg_time = avg_time
                    best_reachable = reachable

            if best_base is None:
                break

            chosen_bases.append(best_base)
            newly_covered = set(best_reachable)
            uncovered -= newly_covered
            covered |= newly_covered

            cumulative_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())
            round_details.append({
                "ronde": ronde,
                "basis_postcode": best_base,
                "nieuwe_postcodes": len(newly_covered),
                "nieuwe_mensen": float(best_people),
                "cumulatief_mensen": cumulative_people,
            })

        total_locations = len(tt.columns)
        covered_locations = len(covered)
        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

        return {
            "strategy": "max_coverage",
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": covered_locations,
            "total_locations": total_locations,
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0,
        }


class AmbulanceOptimizer:
    """
    Vergelijkt verschillende optimalisatie-strategieën:
    1. Max coverage (meeste mensen bereikt)
    2. Max speed (snelste gemiddelde responstijd)
    3. Balanced (mix van beiden)
    """

    def __init__(self, df_traveltimes: pd.DataFrame, df_demand: pd.DataFrame, max_response_time: int = 900):
        self.df_traveltimes = df_traveltimes
        self.df_demand = df_demand
        self.max_response_time = max_response_time

    def strategy_max_coverage(self, num_bases: int) -> dict:
        """Standaard greedy: meeste nieuwe mensen per ronde"""
        optimizer = GreedyOptimizer(max_locations=num_bases, max_response_time=self.max_response_time)
        return optimizer.find_optimal_locations(self.df_traveltimes, self.df_demand)

    def strategy_max_speed(self, num_bases: int) -> dict:
        """Kies bases die de LAAGSTE gemiddelde responstijd hebben voor bereikbare locaties"""
        demand_lookup = (
            self.df_demand.drop_duplicates(subset=["Postal codes"])
            .set_index("Postal codes")["Demand"]
        )

        uncovered = set(self.df_traveltimes.columns)
        chosen_bases = []
        covered = set()
        round_details = []

        for ronde in range(1, num_bases + 1):
            best_base = None
            best_avg_time = float("inf")
            best_people = -1.0
            best_reachable = []

            for base in self.df_traveltimes.index:
                row = self.df_traveltimes.loc[base]
                reachable = [
                    loc for loc in uncovered
                    if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time
                ]

                if len(reachable) == 0:
                    continue

                # Gemiddelde responstijd naar bereikbare locaties
                times = [pd.to_numeric(row[loc], errors="coerce") for loc in reachable]
                avg_time = sum(times) / len(times)
                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())

                # Kies op laagste gemiddelde tijd, tie-break op meeste mensen
                if (avg_time < best_avg_time) or (avg_time == best_avg_time and new_people > best_people):
                    best_base = base
                    best_avg_time = avg_time
                    best_people = new_people
                    best_reachable = reachable

            if best_base is None:
                break

            chosen_bases.append(best_base)
            newly_covered = set(best_reachable)
            uncovered -= newly_covered
            covered |= newly_covered

            cumulative_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

            round_details.append({
                "ronde": ronde,
                "basis_postcode": best_base,
                "gemiddelde_responstijd": best_avg_time,
                "nieuwe_mensen": float(best_people),
                "cumulatief_mensen": cumulative_people,
            })

        total_locations = len(self.df_traveltimes.columns)
        covered_locations = len(covered)
        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

        return {
            "strategy": "max_speed",
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": covered_locations,
            "total_locations": total_locations,
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0,
        }

    def strategy_balanced(self, num_bases: int) -> dict:
        """
        Balans tussen mensen en snelheid:
        Score = (aantal_bereikbare_mensen / max_mensen) * (1 - avg_time / max_response_time)
        """
        demand_lookup = (
            self.df_demand.drop_duplicates(subset=["Postal codes"])
            .set_index("Postal codes")["Demand"]
        )

        uncovered = set(self.df_traveltimes.columns)
        chosen_bases = []
        covered = set()
        round_details = []

        for ronde in range(1, num_bases + 1):
            best_base = None
            best_score = -1.0
            best_people = -1.0
            best_reachable = []
            best_avg_time = 0

            for base in self.df_traveltimes.index:
                row = self.df_traveltimes.loc[base]
                reachable = [
                    loc for loc in uncovered
                    if pd.to_numeric(row[loc], errors="coerce") <= self.max_response_time
                ]

                if len(reachable) == 0:
                    continue

                new_people = float(demand_lookup.reindex(reachable).fillna(0).sum())
                times = [pd.to_numeric(row[loc], errors="coerce") for loc in reachable]
                avg_time = sum(times) / len(times)

                # Genormaliseerde score
                people_score = new_people / max(demand_lookup.values)
                speed_score = 1 - (avg_time / self.max_response_time)
                combined_score = (people_score * 0.6) + (speed_score * 0.4)  # 60% mensen, 40% snelheid

                if combined_score > best_score:
                    best_base = base
                    best_score = combined_score
                    best_people = new_people
                    best_reachable = reachable
                    best_avg_time = avg_time

            if best_base is None:
                break

            chosen_bases.append(best_base)
            newly_covered = set(best_reachable)
            uncovered -= newly_covered
            covered |= newly_covered

            cumulative_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

            round_details.append({
                "ronde": ronde,
                "basis_postcode": best_base,
                "nieuwe_mensen": float(best_people),
                "gemiddelde_responstijd": best_avg_time,
                "cumulatief_mensen": cumulative_people,
            })

        total_locations = len(self.df_traveltimes.columns)
        covered_locations = len(covered)
        total_people = float(demand_lookup.sum())
        covered_people = float(demand_lookup.reindex(list(covered)).fillna(0).sum())

        return {
            "strategy": "balanced",
            "chosen_bases": chosen_bases,
            "round_details": round_details,
            "covered_locations": covered_locations,
            "total_locations": total_locations,
            "covered_people": covered_people,
            "total_people": total_people,
            "covered_people_pct": (covered_people / total_people * 100) if total_people > 0 else 0,
        }

    def compare_strategies(self, num_bases: int = 5) -> pd.DataFrame:
        """Voer alle strategieën uit en vergelijk resultaten"""
        results = []

        print("Berekenen strategieën...\n")

        # Strategy 1: Max Coverage
        print("1. Max Coverage (meeste mensen)...")
        cov = self.strategy_max_coverage(num_bases)
        results.append({
            "Strategie": "Max Coverage",
            "Bases": ", ".join(cov["chosen_bases"]),
            "Bereikt": int(cov["covered_people"]),
            "Percentage": f"{cov['covered_people_pct']:.2f}%",
            "Postcodes": cov["covered_locations"],
        })

        # Strategy 2: Max Speed
        print("2. Max Speed (snelste responstijd)...")
        spd = self.strategy_max_speed(num_bases)
        results.append({
            "Strategie": "Max Speed",
            "Bases": ", ".join(spd["chosen_bases"]),
            "Bereikt": int(spd["covered_people"]),
            "Percentage": f"{spd['covered_people_pct']:.2f}%",
            "Postcodes": spd["covered_locations"],
        })

        # Strategy 3: Balanced
        print("3. Balanced (60% mensen, 40% snelheid)...")
        bal = self.strategy_balanced(num_bases)
        results.append({
            "Strategie": "Balanced",
            "Bases": ", ".join(bal["chosen_bases"]),
            "Bereikt": int(bal["covered_people"]),
            "Percentage": f"{bal['covered_people_pct']:.2f}%",
            "Postcodes": bal["covered_locations"],
        })

        df_comparison = pd.DataFrame(results)
        return df_comparison, {"max_coverage": cov, "max_speed": spd, "balanced": bal}