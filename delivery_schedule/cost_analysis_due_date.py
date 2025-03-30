import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load due-date-based daily volume distribution
df = pd.read_csv("due_date_distribution.csv")

# --- Parameters ---
thresholds = np.arange(600, 1200 + 50, 50)
k_per_m3 = 30
K1 = 262  # 900 ft³ truck cost
K2 = 328  # 1800 ft³ truck cost
c = 0.10 * k_per_m3
holding_cost_per_cuft_per_day = c / 35.3147  # convert from m³ to ft³

# --- Store Results ---
results = []

for threshold in thresholds:
    expected_cost = 0
    expected_shipments = 0
    valid_policy = True

    for _, row in df.iterrows():
        v2, v3, v4, prob = row["Volume_due_in_2"], row["Volume_due_in_3"], row["Volume_due_in_4"], row["Probability"]

        # 1. Age inventory (shift: v3 → v2, v4 → v3, new arrivals added below)
        aged_v2 = v3
        aged_v3 = v4
        aged_v4 = 0  # will be filled by today's new arrivals
        total_volume = v2 + v3 + v4

        # 2. Check if threshold triggers shipment
        if total_volume >= threshold or v2 > 0:
            expected_shipments += prob

            # Ship in FIFO: v2 first, then v3, then v4
            remaining_volume = total_volume
            to_ship = threshold
            ship_v2 = min(v2, to_ship)
            to_ship -= ship_v2
            ship_v3 = min(v3, to_ship)
            to_ship -= ship_v3
            ship_v4 = min(v4, to_ship)

            leftover_v2 = v2 - ship_v2
        else:
            leftover_v2 = v2  # nothing shipped, all v2 left behind

        # 3. Check for lateness
        if leftover_v2 > 0:
            valid_policy = False
            break

        # 4. Compute costs
        holding_cost = (v2 + v3 + v4) * holding_cost_per_cuft_per_day

        if total_volume >= threshold:
            truck_cost = K1 if total_volume <= 900 else K2
            variable_cost = (k_per_m3 / 35.3147) * total_volume
        else:
            truck_cost = 0
            variable_cost = 0

        total_cost = holding_cost + variable_cost + truck_cost
        expected_cost += prob * total_cost

    results.append({
        "Threshold": threshold,
        "Valid": valid_policy,
        "Expected_Daily_Cost": round(expected_cost, 2) if valid_policy else None,
        "Expected_Shipments_per_Day": round(expected_shipments, 4) if valid_policy else None
    })

# --- Output Results ---
df_results = pd.DataFrame(results)
print(df_results)

df_results.to_csv("due_date_threshold_policy_costs.csv", index=False)

# --- Plot valid policies ---
df_valid = df_results[df_results["Valid"] == True]
plt.figure(figsize=(10, 6))
plt.plot(df_valid["Threshold"], df_valid["Expected_Daily_Cost"], marker='o', label="Expected Cost (Valid Policies)")
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Threshold Policy Cost (With Due Date Constraints)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
