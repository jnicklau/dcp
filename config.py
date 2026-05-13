# Configuration for DLA optimization system

# Battery parameters
BATTERY_CAPACITY_KWH = 500.0
BATTERY_MAX_POWER_KW = 100.0
BATTERY_ROUND_TRIP_EFFICIENCY = 0.90

# Optimization
HORIZON_HOURS = 12
OPTIMIZATION_STEPS_PER_HOUR = 4  # 15-minute intervals
N_SCENARIOS = 10
OPT_FREQUENCY = 3*4  # Optimize every 12 hours (48 steps) 

# Model hyperparameters
HIDDEN_DIMS = [64, 64, 32]
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_MC_SAMPLES = 50

# Uncertainty method: 
#   "mcd" = MC Dropout, 
#   "bnn" = Bayes by Backprop (true BNN)
UNCERTAINTY_METHOD = "mcd"


# Data paths
FUNDIUM_DATA_PATH = "fondium_15_min_data_2023.csv"
PRICE_DATA_PATH = (
    "energy-charts_Stromproduktion_und_Börsenstrompreise_in_Deutschland_2023.csv"
)

# Train/test split
TRAIN_SPLIT = 0.8

# Lag steps for abwaerme (waste heat) temporal features: 1hr, 2hr, 3hr, 6hr, 1day
ABWAERME_LAG_STEPS = [48,96,672]
DLA_LAG_STEPS = [48,96,672]
PRICE_LAG_STEPS = [48,96,672]

# print a warning if lag steps are too small
# compared to the optimization horizon (48 steps for 12 hours at 15-minute intervals) 

if any(step < HORIZON_HOURS * OPTIMIZATION_STEPS_PER_HOUR for step in ABWAERME_LAG_STEPS):
    print("Warning: abwaerme lag steps are smaller than the optimization horizon. Data leakage may occur.")

if any(step < HORIZON_HOURS * OPTIMIZATION_STEPS_PER_HOUR for step in DLA_LAG_STEPS):
    print("Warning: DLA lag steps are smaller than the optimization horizon. Data leakage may occur.")

if any(step < HORIZON_HOURS * OPTIMIZATION_STEPS_PER_HOUR for step in PRICE_LAG_STEPS):
    print("Warning: Price lag steps are smaller than the optimization horizon. Data leakage may occur.")
