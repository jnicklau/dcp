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
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
NUM_MC_SAMPLES = 50  # For aleatoric uncertainty

# Data paths
FUNDIUM_DATA_PATH = "fondium_15_min_data_2023.csv"
PRICE_DATA_PATH = (
    "energy-charts_Stromproduktion_und_Börsenstrompreise_in_Deutschland_2023.csv"
)

# Train/test split
TRAIN_SPLIT = 0.8

# Lag steps for abwaerme (waste heat) temporal features: 15min, 1hr, 24hr, 1wk
ABWAERME_LAG_STEPS = [4, 8, 12, 24, 96]
