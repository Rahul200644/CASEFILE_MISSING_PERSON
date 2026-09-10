import numpy as np
import pandas as pd

class LocationMarkovChain:
    """
    First-order Markov Chain model to learn transition probabilities between geographical areas.
    """
    def __init__(self):
        self.transitions = {}
        self.transition_matrix = None
        self.states = []

    def fit(self, location_sequences):
        """
        location_sequences: list of lists or pandas series of location names/IDs
        """
        counts = {}
        all_states = set()
        
        for seq in location_sequences:
            for i in range(len(seq) - 1):
                src, dst = seq[i], seq[i+1]
                all_states.add(src)
                all_states.add(dst)
                if src not in counts:
                    counts[src] = {}
                counts[src][dst] = counts[src].get(dst, 0) + 1

        self.states = sorted(list(all_states))
        self.transitions = {}
        
        for src in self.states:
            self.transitions[src] = {}
            total = sum(counts.get(src, {}).values())
            if total > 0:
                for dst in self.states:
                    self.transitions[src][dst] = counts.get(src, {}).get(dst, 0) / total
            else:
                for dst in self.states:
                    self.transitions[src][dst] = 1.0 / len(self.states) if self.states else 0.0

        # Build transition matrix dataframe
        matrix_data = []
        for src in self.states:
            row = [self.transitions[src].get(dst, 0.0) for dst in self.states]
            matrix_data.append(row)
        
        self.transition_matrix = pd.DataFrame(matrix_data, index=self.states, columns=self.states)

    def predict_next_steps(self, current_location, max_steps=4):
        """
        Generate probable route sequence starting from current_location.
        """
        if current_location not in self.states:
            if self.states:
                current_location = self.states[0]
            else:
                return [current_location], 1.0

        route = [current_location]
        curr = current_location
        prob_accum = 1.0

        for _ in range(max_steps - 1):
            if curr not in self.transitions or not self.transitions[curr]:
                break
            next_probs = self.transitions[curr]
            # Exclude self-transition if possible to get movement route
            candidates = {k: v for k, v in next_probs.items() if k != curr and v > 0}
            if not candidates:
                candidates = next_probs
            
            best_next = max(candidates.items(), key=lambda x: x[1])[0]
            step_prob = candidates[best_next]
            if step_prob == 0:
                break
            route.append(best_next)
            prob_accum *= step_prob
            curr = best_next

        return route, float(prob_accum)

    def get_route_string(self, current_location, max_steps=4):
        route, prob = self.predict_next_steps(current_location, max_steps)
        return " → ".join(route), prob
