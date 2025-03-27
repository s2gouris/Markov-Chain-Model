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

# --- Step 1: Daily Production Volume Distribution ---
volume_dist = defaultdict(float)
component_list = list(component_volumes.keys())

# Poisson probabilities for 0 to 5 orders (rest lumped into 6+)
poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(6)]
poisson_probs.append(1 - sum(poisson_probs))  # 6+ bucket

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

# --- Step 2: Markov Chain States (Volume only) ---
states = [v for v in range(0, max_volume + 1, volume_step)]
state_index = {v: i for i, v in enumerate(states)}
n_states = len(states)

# --- Step 3: One-Day Transition Matrix ---
transition_matrix = np.zeros((n_states, n_states))

for v in states:
    i = state_index[v]
    for added_v, prob in volume_dist.items():
        next_v = min(v + added_v, max_volume)
        j = state_index[next_v]
        transition_matrix[i, j] += prob

# --- Step 4: Save to CSV ---
df = pd.DataFrame(transition_matrix, index=states, columns=states)
df.to_csv("one_day_transition_matrix.csv")
print("Transition matrix saved as 'one_day_transition_matrix.csv'.")

# --- Step 5: Visualize Transition Probabilities ---
plt.figure(figsize=(14, 8))
sns.heatmap(df, cmap="Blues", cbar=True)
plt.title("One-Day Volume Transition Matrix")
plt.xlabel("Next Volume (ft³)")
plt.ylabel("Current Volume (ft³)")
plt.tight_layout()
plt.show()

# --- Step 6: Cost Parameters and Calculation ---
k = 120        # Variable cost per m³
c = 0.1 * k    # Holding cost per m³/day (≥10% of k)
K1 = 950       # 900 ft³ truck
K2 = 1500      # 1800 ft³ truck

# Expected shipment volume over 4-day cycle
expected_daily_volume = sum(vol * prob for vol, prob in volume_dist.items())
expected_shipment_volume = expected_daily_volume * 4
shipment_volume_m3 = expected_shipment_volume / 35.315  # Convert ft³ to m³

# 3PL Cost
fixed_cost = 800
variable_cost = k * shipment_volume_m3
holding_cost = c * (shipment_volume_m3 * (1 + 2 + 3) / 4)  # Average days held: 1.5
total_3pl_cost = fixed_cost + variable_cost + holding_cost
average_3pl_per_day = total_3pl_cost / 4

# Truck Rental Cost
if expected_shipment_volume <= 900:
    truck_cost = K1
elif expected_shipment_volume <= 1800:
    truck_cost = K2
else:
    truck_cost = float("inf")
average_truck_per_day = truck_cost / 4

# Output Costs
print(f"Average 3PL Cost/Day: ${average_3pl_per_day:.2f}")
print(f"Average Truck Rental Cost/Day: ${average_truck_per_day:.2f}")