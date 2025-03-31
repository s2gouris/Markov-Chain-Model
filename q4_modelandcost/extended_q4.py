import numpy as np
import pandas as pd
import itertools
import math
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
lambda_orders = 2
max_daily_production = 3
volume_step = 50
max_volume = 1800
max_due_days = 5  # Maximum days an order can be due before being late

def generate_poisson_distribution(lambda_orders, max_orders=5):
    poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(max_orders)]
    poisson_probs.append(1 - sum(poisson_probs))  # Bucket for 6+ orders
    return poisson_probs

# --- Step 1: Daily Volume Distribution ---
volume_dist = defaultdict(float)
component_list = list(component_volumes.keys())
poisson_probs = generate_poisson_distribution(lambda_orders)

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

# --- Step 2: Define Extended States (Volume, Due Date) ---
volume_states = list(range(0, max_volume + volume_step, volume_step))  # 0, 50, ..., 1800
due_states = list(range(max_due_days + 1))  # 0, 1, ..., max_due_days
state_space = list(itertools.product(volume_states, due_states))
state_index = {state: i for i, state in enumerate(state_space)}
n_states = len(state_space)

# --- Step 3: Build Extended Transition Matrix ---
P = np.zeros((n_states, n_states))

for (v, due), i in state_index.items():
    for added_v, prob in volume_dist.items():
        new_total = v + added_v
        new_due = max(due - 1, 0)  # Decrease due date (closer to late)

        if new_total > max_volume:
            # Ship 1800 ft³, carry over remainder
            carryover = new_total - 1800
            carryover = min(carryover, max_volume)
            new_state = (carryover, new_due)
        else:
            new_state = (new_total, new_due)
        
        j = state_index[new_state]
        P[i, j] += prob

# --- Step 4: Save Extended Transition Matrix ---
df = pd.DataFrame(P, index=state_space, columns=state_space)
df.to_csv("extended_transition_matrix.csv")
print("Extended transition matrix saved successfully.")

# --- Step 5: Visualize with a Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(df, cmap="YlGnBu", annot=False, fmt=".2f", cbar=True)
plt.title("Extended Transition Matrix Heatmap")
plt.xlabel("New State (Volume, Due Date)")
plt.ylabel("Current State (Volume, Due Date)")
plt.show()
