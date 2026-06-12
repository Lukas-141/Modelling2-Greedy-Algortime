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

    def _vraag(self) -> pd.Series:
        vraag = self.df_demand.drop_duplicates(subset=["Postal codes"])
        return vraag.set_index("Postal codes")["Demand"]

    def _bereikbaar(self, row: pd.Series, open_postcodes):
        return [postcode for postcode in open_postcodes if pd.to_numeric(row[postcode], errors="coerce") <= self.max_response_time]

    def _loop(self, num_bases: int, strategy: str, score_fn):
        vraag = self._vraag()
        open_postcodes = set(self.df_traveltimes.columns)
        bases, gedekt, rondes = [], set(), []

        for ronde in range(1, num_bases + 1):
            beste = None

            for base in self.df_traveltimes.index:
                row = self.df_traveltimes.loc[base]
                bereikbaar = self._bereikbaar(row, open_postcodes)
                if not bereikbaar:
                    continue

                nieuwe_mensen = float(vraag.reindex(bereikbaar).fillna(0).sum())
                gemiddelde_tijd = float(pd.to_numeric(row[bereikbaar], errors="coerce").mean())
                score = score_fn(nieuwe_mensen, gemiddelde_tijd, bereikbaar, vraag, open_postcodes)

                if beste is None or score > beste["score"]:
                    beste = {"base": base, "reachable": bereikbaar, "people": nieuwe_mensen, "avg_time": gemiddelde_tijd, "score": score}

            if beste is None:
                break

            bases.append(beste["base"])
            nieuw_gedekt = set(beste["reachable"])
            open_postcodes -= nieuw_gedekt
            gedekt |= nieuw_gedekt
            rondes.append({
                "ronde": ronde,
                "basis_postcode": beste["base"],
                "nieuwe_mensen": beste["people"],
                "gemiddelde_responstijd": beste["avg_time"],
                "score": beste["score"],
                "cumulatief_mensen": float(vraag.reindex(list(gedekt)).fillna(0).sum()),
            })

        totaal_mensen = float(vraag.sum())
        gedekte_mensen = float(vraag.reindex(list(gedekt)).fillna(0).sum())
        return {
            "strategy": strategy,
            "chosen_bases": bases,
            "round_details": rondes,
            "covered_locations": len(gedekt),
            "total_locations": len(self.df_traveltimes.columns),
            "covered_people": gedekte_mensen,
            "total_people": totaal_mensen,
            "covered_people_pct": (gedekte_mensen / totaal_mensen * 100) if totaal_mensen > 0 else 0.0,
        }

    def strategy_max_coverage(self, num_bases: int) -> dict:
        return GreedyOptimizer(max_locations=num_bases, max_response_time=self.max_response_time).find_optimal_locations(
            self.df_traveltimes,
            self.df_demand,
        )

    def strategy_max_speed(self, num_bases: int) -> dict:
        return self._loop(num_bases, "max_speed", lambda people, avg_time, reachable, vraag, open_postcodes: (-avg_time, people))

    def strategy_balanced(self, num_bases: int, w_people: float = 0.8, w_speed: float = 0.2) -> dict:
        total_weight = w_people + w_speed
        if total_weight <= 0:
            raise ValueError("w_people + w_speed moet > 0 zijn.")

        w_people /= total_weight
        w_speed /= total_weight

        def score_fn(people, avg_time, reachable, vraag, open_postcodes):
            available = max(float(vraag.reindex(list(open_postcodes)).fillna(0).sum()), 1.0)
            people_score = people / available
            speed_score = 1.0 - min(avg_time / self.max_response_time, 1.0)
            return (w_people * people_score) + (w_speed * speed_score)

        result = self._loop(num_bases, "balanced", score_fn)
        result["w_people"] = w_people
        result["w_speed"] = w_speed
        return result

    def compare_strategies(self, num_bases: int = 5):
        cov = self.strategy_max_coverage(num_bases)
        spd = self.strategy_max_speed(num_bases)
        bal = self.strategy_balanced(num_bases)

        comparison = pd.DataFrame([
            {"Strategie": "Max Coverage", "Bases": ", ".join(cov["chosen_bases"]), "Bereikt": int(cov["covered_people"]), "Percentage": f"{cov['covered_people_pct']:.2f}%", "Postcodes": cov["covered_locations"]},
            {"Strategie": "Max Speed", "Bases": ", ".join(spd["chosen_bases"]), "Bereikt": int(spd["covered_people"]), "Percentage": f"{spd['covered_people_pct']:.2f}%", "Postcodes": spd["covered_locations"]},
            {"Strategie": "Balanced", "Bases": ", ".join(bal["chosen_bases"]), "Bereikt": int(bal["covered_people"]), "Percentage": f"{bal['covered_people_pct']:.2f}%", "Postcodes": bal["covered_locations"]},
        ])
        return comparison, {"max_coverage": cov, "max_speed": spd, "balanced": bal}

    def scan_balanced_weights(self, num_bases: int = 5, step: float = 0.05):
        if step <= 0 or step > 1:
            raise ValueError("step moet in (0, 1] liggen.")

        rows, results = [], {}
        steps = int(round(1.0 / step))

        for index in range(steps + 1):
            w_people = round(index * step, 10)
            w_speed = round(1.0 - w_people, 10)
            result = self.strategy_balanced(num_bases=num_bases, w_people=w_people, w_speed=w_speed)
            results[f"{w_people:.2f}_{w_speed:.2f}"] = result

            mean_rt = pd.DataFrame(result["round_details"])["gemiddelde_responstijd"].mean() if result["round_details"] else None
            rows.append({
                "w_people": round(w_people, 2),
                "w_speed": round(w_speed, 2),
                "bereikte_mensen": int(result["covered_people"]),
                "bereik_pct": round(result["covered_people_pct"], 2),
                "gedekte_postcodes": int(result["covered_locations"]),
                "gem_resp_tijd_sec": round(float(mean_rt), 2) if mean_rt is not None else None,
                "bases": ", ".join(result["chosen_bases"]),
            })

        weights = pd.DataFrame(rows).sort_values(
            by=["bereikte_mensen", "gedekte_postcodes", "gem_resp_tijd_sec"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        return weights, results
