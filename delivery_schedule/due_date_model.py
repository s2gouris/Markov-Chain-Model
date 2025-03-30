import itertools
import numpy as np
from collections import defaultdict
import math
import pandas as pd

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
due_date_probs = {2: 0.3, 3: 0.4, 4: 0.3}

lambda_orders = 2
max_daily_production = 3

# --- Generate All Order Combinations ---
component_list = list(component_volumes.keys())
poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(6)]
poisson_probs.append(1 - sum(poisson_probs))  # Probability for 6+ orders

# --- Create Distribution Over (v2, v3, v4) ---
# A dictionary where keys are (v2, v3, v4) and values are probabilities
due_date_volume_dist = defaultdict(float)

for n_orders, p_orders in enumerate(poisson_probs):
    actual_orders = min(n_orders, max_daily_production)
    if actual_orders == 0:
        due_date_volume_dist[(0, 0, 0)] += p_orders
        continue

    combos = list(itertools.product(component_list, repeat=actual_orders))
    
    for combo in combos:
        combo_prob = p_orders
        for c in component_list:
            combo_prob *= component_probs[c] ** combo.count(c)

        # Volume per order
        volumes = [component_volumes[c] for c in combo]

        # Now randomly assign each volume to a due date
        due_date_assignments = list(itertools.product([2, 3, 4], repeat=actual_orders))
        
        for assignment in due_date_assignments:
            assign_prob = combo_prob
            v2 = v3 = v4 = 0

            for vol, dd in zip(volumes, assignment):
                assign_prob *= due_date_probs[dd]
                if dd == 2:
                    v2 += vol
                elif dd == 3:
                    v3 += vol
                elif dd == 4:
                    v4 += vol

            due_date_volume_dist[(v2, v3, v4)] += assign_prob

# --- Display or Export the Distribution ---
print("Sample of daily due-date-based volume distribution:")
for state, prob in list(due_date_volume_dist.items())[:10]:
    print(f"{state}: {round(prob, 5)}")

# Save to CSV or pickle for use in next step
rows = []
for (v2, v3, v4), prob in due_date_volume_dist.items():
    rows.append({"Volume_due_in_2": v2, "Volume_due_in_3": v3, "Volume_due_in_4": v4, "Probability": prob})

df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("due_date_distribution.csv", index=False)
print("\nSaved to due_date_volume_distribution.csv")