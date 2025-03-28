import numpy as np

# --- Parameters ---
days_to_simulate = 1000
arrival_rate = 2  # orders per day
production_capacity = 3  # max per day
shipment_threshold = 900  # ft³

# Product info
products = {
    'A': {'prob': 0.3, 'volume': 50},
    'B': {'prob': 0.5, 'volume': 100},
    'C': {'prob': 0.2, 'volume': 150}
}
product_types = list(products.keys())
product_probs = [products[p]['prob'] for p in product_types]

# --- Cost Parameters (DUMMY) ---
# 3PL Option
fixed_cost_3pl = 800
k = 2.0  # per m³ (convert ft³ to m³ = divide by 35.315)
c = 0.20  # per day per m³

# Truck Option
K1 = 1000  # 900 ft³ truck
K2 = 1600  # 1800 ft³ truck

# Choose logistics mode
use_3pl = True  # Set False to use truck rental

# --- Tracking ---
inventory = []
inventory_volume = 0
total_holding_cost = 0
total_shipping_cost = 0
shipments = []

# --- Simulation Loop ---
for day in range(days_to_simulate):
    # 1. Order arrivals
    orders_today = min(np.random.poisson(arrival_rate), production_capacity)
    todays_orders = list(np.random.choice(product_types, size=orders_today, p=product_probs))

    # 2. Add to inventory
    for item in todays_orders:
        inventory.append(item)
        inventory_volume += products[item]['volume']

    # 3. Holding cost for the day
    inv_m3 = inventory_volume / 35.315  # ft³ to m³
    total_holding_cost += c * inv_m3

    # 4. Shipment trigger
    if inventory_volume >= shipment_threshold:
        shipment_m3 = inventory_volume / 35.315

        if use_3pl:
            shipping_cost = fixed_cost_3pl + k * shipment_m3
        else:
            if inventory_volume <= 900:
                shipping_cost = K1
            elif inventory_volume <= 1800:
                shipping_cost = K2
            else:
                shipping_cost = K2 + ((inventory_volume - 1800) / 100) * 100  # dummy extra fee

        total_shipping_cost += shipping_cost

        shipments.append({
            'day': day,
            'items': inventory.copy(),
            'volume': inventory_volume,
            'cost': shipping_cost
        })

        inventory.clear()
        inventory_volume = 0

# --- Results ---
total_cost = total_holding_cost + total_shipping_cost
avg_cost_per_day = total_cost / days_to_simulate
avg_shipment_cost = total_shipping_cost / len(shipments) if shipments else 0

print("Simulation Complete.")
print(f"Days Simulated: {days_to_simulate}")
print(f"Shipments Made: {len(shipments)}")
print(f"Total Holding Cost: ${total_holding_cost:.2f}")
print(f"Total Shipping Cost: ${total_shipping_cost:.2f}")
print(f"Total Cost: ${total_cost:.2f}")
print(f"Average Daily Cost: ${avg_cost_per_day:.2f}")
print(f"Average Shipment Cost: ${avg_shipment_cost:.2f}")
print(f"Logistics Option: {'3-PL' if use_3pl else 'Truck Rental'}")
