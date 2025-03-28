import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
lambda_orders = 2
max_daily_production = 3
volume_step = 50
max_volume = 1800

# --- Step 1: Daily Production Probabilities ---
volume_dist = defaultdict(float)
component_list = list(component_volumes.keys())

# Poisson distribution for 0 to 5 orders, rest bucketed as 6+
poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(6)]
poisson_probs.append(1 - sum(poisson_probs))  # probability for 6+ orders

for n_orders, p in enumerate(poisson_probs):
    actual_orders = min(n_orders, max_daily_production)
    combos = list(itertools.product(component_list, repeat=actual_orders))
    for combo in combos:
        vol = sum(component_volumes[c] for c in combo)
        prob = p
        for c in component_list:
            if actual_orders > 0:
                prob *= (component_probs[c] ** combo.count(c))
        volume_dist[vol] += prob

# --- Step 2: Define Markov States ---
states = [(v, d) for v in range(0, max_volume + 1, volume_step) for d in range(1, 5)]
state_index = {s: i for i, s in enumerate(states)}
n_states = len(states)

# --- Step 3: Build Transition Matrix ---
transition_matrix = np.zeros((n_states, n_states))

for (v, d) in states:
    i = state_index[(v, d)]
    if d < 4:
        for added_v, prob in volume_dist.items():
            next_v = min(v + added_v, max_volume)
            next_state = (next_v, d + 1)
            j = state_index[next_state]
            transition_matrix[i, j] += prob
    else:
        # Day 4: reset to (0,1)
        j = state_index[(0, 1)]
        transition_matrix[i, j] = 1.0

# --- Step 4: Save Transition Matrix to CSV ---
df = pd.DataFrame(transition_matrix, index=[str(s) for s in states], columns=[str(s) for s in states])
df.to_csv("acme_transition_matrix.csv")
print("Transition matrix saved as 'acme_transition_matrix.csv'.")

# --- Optional Step 5: Visualize a subset ---
#subset_rows = [s for s in states if s[1] == 1]
#subset_cols = [s for s in states if s[1] == 2]
#subset_df = df.loc[[str(s) for s in subset_rows], [str(s) for s in subset_cols]]

#plt.figure(figsize=(14, 8))
#sns.heatmap(subset_df, cmap="Blues", cbar=True)
#plt.title("Transition Probabilities from Day 1 to Day 2 (subset)")
#plt.xlabel("To State")
#plt.ylabel("From State")
#plt.tight_layout()
#plt.show()

# Parameters - research this better
k = 120  # variable cost per m³
c = 0.1 * k  # must be ≥ 10% of k
K1 = 950  # 900 ft³ truck
K2 = 1500  # 1800 ft³ truck

# Step 1: Expected volume
expected_daily_volume = sum(vol * prob for vol, prob in volume_dist.items())
expected_shipment_volume = expected_daily_volume * 4  # 4-day cycle
shipment_volume_m3 = expected_shipment_volume / 35.315

# 3PL Cost
fixed_cost = 800
variable_cost = k * shipment_volume_m3
holding_cost = c * (shipment_volume_m3 * (1 + 2 + 3) / 4)

total_3pl_cost = fixed_cost + variable_cost + holding_cost
average_3pl_per_day = total_3pl_cost / 4

# Truck Rental
if expected_shipment_volume <= 900:
    truck_cost = K1
elif expected_shipment_volume <= 1800:
    truck_cost = K2
else:
    truck_cost = float("inf")

average_truck_per_day = truck_cost / 4

# Compare
print(f"Average 3PL Cost/Day: ${average_3pl_per_day:.2f}")
print(f"Average Truck Rental Cost/Day: ${average_truck_per_day:.2f}")

