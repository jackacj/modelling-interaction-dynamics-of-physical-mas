# Constant Hyperparameters (For Clarity & Ease of Editing)
# Roles are Hardcoded and Shared Across Scenarios - Role: One-Hot-Encoding Index
"""
roleDict = {
    "defaultAgent": 0,
    "blueTeamAgent": 1,
    "redTeamAgent": 2,
    "tagPredator": 3,
    "tagPrey": 4,
    "flockLeader": 5,
    "physicsBall": 6,
    "physicsBox": 7,
    "obstacle": 8,
    "goalNavigation": 9,
    "goalTransport": 10
}
"""
NUM_ROLES = 11
FEATURE_COLUMNS = [f"agentRole{i}" for i in range(NUM_ROLES)] + ["velX", "velY", "heading"]
DATA_COLUMNS = ["sceneId", "timestep", "agentId", "posX", "posY"] + FEATURE_COLUMNS

# Simulation Hyperparemeters
NUM_ENVS = 60
NUM_STEPS = 250
NUM_AGENTS = 10 # Must Be Even
# Available Scenarios - "flocking", "football", "navigation", "simple_tag", "transport"
# SCENARIO = "flocking" # Deprecated

# Data Hyperparameters
# HISTORY = 5 # Deprecated
# HORIZON = 5 # Deprecated
# K_NEAREST = 2 # Deprecated
AGENT_MIN = 2
TRAINING_RATIO = 0.8

# Model Building/Training Hyperparameters
ENCODING_DIM = 128
NUM_EPOCHS = 30
BATCH_SIZE = 16
