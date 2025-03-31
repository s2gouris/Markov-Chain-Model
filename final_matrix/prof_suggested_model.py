import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
lambda_orders = 2
max_daily_production = 3
volume_step = 50
max_volume = 1800

# --- Step 1: Daily Volume Distribution ---
volume_dist = defaultdict(float)
component_list = list(component_volumes.keys())

# Poisson distribution for 0 to 5 orders
poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(6)]
poisson_probs.append(1 - sum(poisson_probs))  # bucket for 6+ orders

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

# --- Step 2: Define States (Volume only) ---
volume_states = list(range(0, max_volume + volume_step, volume_step))  # 0, 50, ..., 1800
state_index = {v: i for i, v in enumerate(volume_states)}
n_states = len(volume_states)

# --- Step 3: Build One-Day Transition Matrix ---
P = np.zeros((n_states, n_states))

for v in volume_states:
    i = state_index[v]
    for added_v, prob in volume_dist.items():
        new_total = v + added_v
        if new_total <= max_volume:
            j = state_index[new_total]
        else:
            # Ship 1800, carry over remaining
            carryover = new_total - 1800
            carryover = min(carryover, max_volume)  # cap it to max_volume
            j = state_index[carryover]
        P[i, j] += prob

# --- Step 4: Save to CSV ---
#df = pd.DataFrame(P, index=volume_states, columns=volume_states)
#df.to_csv("prof_suggested_model.csv")
#print("Transition matrix saved")

# --- Step 5: Visualize Transition Probabilities ---
#plt.figure(figsize=(14, 8))
#sns.heatmap(df, cmap="Blues", cbar=True)
#plt.title("One-Day Volume Transition Matrix")
#plt.xlabel("Next Volume (ft³)")
#plt.ylabel("Current Volume (ft³)")
#plt.tight_layout()
#plt.show()
