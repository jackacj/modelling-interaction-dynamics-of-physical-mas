# Imports

import csv
import cv2
import torch
import numpy as np
import vmas
import json
import os
from tqdm import tqdm
import hyperparameters
import argparse
from datetime import datetime

### HELPER FUNCTIONS

def makeID(stage, tag):
    now = datetime.now()
    return f"{stage}_{now:%Y%m%d_%H%M}_{tag}"

def renderEnvGrid(env, envIds=(0,1,2,3), scale=0.6, cols=2, windowName="VMAS Grid"):
    frames = []

    # Render Indiviual Sim Frames
    for envId in envIds:
        frame = env.render(mode="rgb_array", env_index=envId)
        frames.append(frame)

    h, w, c = frames[0].shape
    rows = int(np.ceil(len(frames) / cols))

    canvas = np.zeros((rows*h, cols*w, c), dtype=frames[0].dtype)

    # Construct Window Image
    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = frame

    # Scale Window Image
    if scale != 1.0:
        canvas = cv2.resize(
            canvas,
            (int(canvas.shape[1]*scale), int(canvas.shape[0]*scale)),
            interpolation=cv2.INTER_AREA
        )

    # Convert RGB → BGR for OpenCV
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imshow(windowName, canvas)
    cv2.waitKey(1)

# Calculate the Boundary Repulsion Force between Agents and Landmarks - Interaction Dynamics
# boundaryMargin: How close Agents must be before experiencing a repulsive force
# boundaryGain: strength of that repulsive force 
def calculateBoundaryRepulsion(positions, boundaryMargin, boundaryGain):
    # Calculate Boundary Repulsion Force
    distanceLeft   = positions[..., 0] + 1
    distanceRight  = 1 - positions[..., 0]
    distanceBottom = positions[..., 1] + 1
    distanceTop    = 1 - positions[..., 1]
    margin = 1 - boundaryMargin

    repulseX = torch.zeros_like(distanceLeft)
    repulseY = torch.zeros_like(distanceBottom)

    repulseX += (distanceLeft < margin) * (margin - distanceLeft)
    repulseX -= (distanceRight < margin) * (margin - distanceRight)

    repulseY += (distanceBottom < margin) * (margin - distanceBottom)
    repulseY -= (distanceTop < margin) * (margin - distanceTop)

    # Return Boundary Repulsion Force
    return boundaryGain * torch.stack([repulseX, repulseY], dim=-1)

# AI Generated Function - Used for Debugging Simulations by Dumping Data
def describe_entities(env):

    def tensor_to_str(x):
        if isinstance(x, torch.Tensor):
            if x.numel() <= 6:
                return x.detach().cpu().numpy()
            return f"Tensor(shape={tuple(x.shape)})"
        return x

    def describe(obj, name):
        print(f"\n{name}: {obj}")
        print("-" * 60)
        for attr in sorted(dir(obj)):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue

            # Skip methods
            if callable(val):
                continue

            val = tensor_to_str(val)
            print(f"{attr:20} : {val}")

    print("\n================ AGENTS ================")
    for i, agent in enumerate(env.world.agents):
        describe(agent, f"Agent[{i}]")

        if hasattr(agent, "state"):
            describe(agent.state, f"Agent[{i}].state")

    print("\n============== LANDMARKS ==============")
    for i, lm in enumerate(env.world.landmarks):
        describe(lm, f"Landmark[{i}]")

        if hasattr(lm, "state"):
            describe(lm.state, f"Landmark[{i}].state")

### SCENARIO CONTROLLERS

# Flocking Behaviour for 'flocking' scenario - AI Assisted
# Agents move as a flock, maintaining degrees of alignment, cohesion & separation
def flockingController(env):
    # Define Scenario Attributes
    numEnvs = env.num_envs
    numAgents = env.n_agents
    actions = torch.zeros(numEnvs, numAgents, 2, device = env.device)
    
    # Positions & Velocities for Every Agent (in Every Sim) - Envs * Agents * 2
    positions = torch.stack([a.state.pos for a in env.world.agents], dim = 1)
    velocities = torch.stack([a.state.vel for a in env.world.agents], dim = 1)

    # Loop through Every Agent
    for i in range(numAgents):
        # Get a Single Agent's Position & Velocity (in Every Sim)
        iPos = positions[:, i:i+1, :]

        # Calculate Difference to Every Other Agent - Envs * Agents * 2
        difference = positions - iPos

        # Calculate Distance to Every Other Agent - Additinal term to prevent zero-div
        distance = torch.norm(difference, dim = -1, keepdim = True) + 1e-6

        # Remove Self-Interaction
        mask = torch.ones_like(distance)
        mask[:, i, :] = 0.0
        difference = difference * mask
        distance = (distance * mask) + 1e-6

        # Calculate Alignment - Sum of Neighbours' Velocities
        # Calulate Cohesion - Sum of Vectors towards Neighbours
        # Calculate Separation - Repulsion Inversely Proportioned to Squared Distance
        alignment = (velocities * mask).sum(dim = 1)
        cohesion = difference.sum(dim = 1)
        separation = -(difference / (distance ** 2)).sum(dim = 1)

        # Set the Agent's Actions - Weighted Sum of Above Three
        actions[:, i] = 0.5*alignment + 0.05*cohesion + 0.2*separation

        # Clamp Actions to <= Unit Magnitude 1.0
        norm = torch.norm(actions, dim=-1, keepdim=True) + 1e-6
        actions = actions / torch.clamp(norm, min=1.0)

    # Convert [numEnvs, numAgents, 2] -> list[numAgents] of [numEnvs, 2]
    actionsList = [actions[:, i, :] for i in range(numAgents)]
    return actionsList

# Vectorised VMAS Port of original MPE 'navigation' controller with extra features - AI Assisted
# Agents navigate to their goals while avoiding other agents & obstacles
# separationGain: the strength of the repulsive force between Agents - Interaction Dynamics
def navigationController(env, separationGain=0.1, boundaryMargin=0.9, boundaryGain=0.05):
    # Define Scenario Attributes
    device = env.device
    numEnvs = env.num_envs
    numAgents = env.n_agents
    actions = torch.zeros(numEnvs, numAgents, 2, device=device)

    # Extract Agents, Goals and their Positions - Env * Agents * 2
    agents = env.world.agents
    positions = torch.stack([a.state.pos for a in agents], dim = 1) 
    goals = torch.stack([a.goal.state.pos for a in agents], dim = 1) 

    # Compute Vector towards an Agent's Goal - Env * Agents * 2
    goalVector = goals - positions  # (E, n_agents, 2)

    # Calculate Inter-Agent Separation - Env * Agents * 2
    difference = positions.unsqueeze(2) - positions.unsqueeze(1)
    distance = torch.norm(difference, dim =- 1, keepdim = True) + 1e-6
    mask = 1 - torch.eye(numAgents, device = device).unsqueeze(0)
    mask = mask.unsqueeze(-1)
    separation = ((difference / (distance ** 2)) * mask).sum(dim = 2)
    
    # Calculate Boundary Repulsion Force
    boundaryForce = calculateBoundaryRepulsion(positions, boundaryMargin, boundaryGain)

    # Combine vectors
    actions = goalVector + separationGain * separation + boundaryForce

    # Clamp Actions to <= Unit Magnitude 1.0
    actions = torch.clamp(actions, -1.0, 1.0)
    return actions.transpose(0,1)

# Vectorised VMAS Port of original MPE 'transport' controller with extra features - AI Assisted
# Agents navigate to the object, position themselves and then push the object to the goal
def transportController(env, separationGain=1e-4, pushStrength=100.0, pickupRadius=0.1, boundaryMargin=0.9, boundaryGain=1.0):
    # Define Scenario Attributes
    device = env.device
    numEnvs = env.num_envs
    numAgents = env.n_agents
    actions = torch.zeros(numEnvs, numAgents, 2, device=device)

    # Gather Agents & Positions
    agents = env.world.agents
    positions = torch.stack([a.state.pos for a in agents], dim=1)

    # Gather the Physics Object & Goal Positions (Set Mass of Object so Pushable)
    obj = env.world.landmarks[1]
    goal = env.world.landmarks[0]
    objPos = obj.state.pos
    goalPos = goal.state.pos
    obj.mass = 1.0

    # Calculate Vector from Object to Goal & Extract Direction
    objToGoal = goalPos - objPos
    objToGoalDir = objToGoal / (torch.norm(objToGoal, dim = -1, keepdim = True) + 1e-6)

    # Calculate Vector from Object to Agent and determine if Agent is Infront
    objToAgent = positions - objPos.unsqueeze(1)
    objToAgentDistance = torch.norm(objToAgent, dim = -1, keepdim = True) + 1e-6
    objToAgentDir = objToAgent / objToAgentDistance
    objToGoalDirTemp = objToGoalDir.unsqueeze(1)
    front = (objToAgentDir * objToGoalDirTemp).sum(dim = -1, keepdim = True) > 0

    # Create Action for Orbiting the Physics Object
    perp = torch.stack([-objToAgentDir[...,1], objToAgentDir[...,0]], dim = -1)
    side = torch.sign((objToGoalDirTemp[...,0] * perp[...,1] - objToGoalDirTemp[...,1] * perp[...,0]).unsqueeze(-1))
    orbit = perp * side

    # Find Desired Pushing Point -> Behind Physics Object in Direction of Goal
    objRadius = obj.shape.length
    agentRadius = agents[0].shape.radius
    penetration = 0.01
    pushOffset = objRadius + agentRadius - penetration
    pushPoint = objPos - pushOffset * objToGoalDir

    # Communicate Pushing Point to All Agents (Vectorisation) - Env * Agents * 2
    pushPoint = pushPoint.unsqueeze(1).expand(-1, numAgents, -1)

    # Distance to Pushing Point
    toPush = pushPoint - positions
    distToPush = torch.norm(toPush, dim = -1, keepdim = True)

    # Create Action for Agent's Near Pushing Point -> Push Physics Object on Vector between Object & Goal
    pushAction = objToGoalDir.unsqueeze(1).expand(-1, numAgents, -1)

    # Scale Orbit by Distance to Push Point & Add Inward Bias
    orbit = orbit * torch.clamp(distToPush / 0.3, 0.0, 1.0)
    inward = -objToAgentDir * 0.1
    orbit = orbit + inward

    # Smoothly Blend between Navigating to Push Point & Pushing Object
    nearPush = (distToPush < pickupRadius).float()
    inward = (objPos.unsqueeze(1) - positions)
    inward = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-6)
    pushVec = pushStrength * (pushAction + 0.3 * inward)

    # Combine All Actions - Navigating to Push Point, Pushing Object & Orbiting
    rear = ~front
    action = (
        rear.float() * ((1 - nearPush) * toPush + nearPush * pushVec)
        + front.float() * orbit
    )

    # Calculate Inter-Agent Separation
    difference = positions.unsqueeze(2) - positions.unsqueeze(1)
    distance = torch.norm(difference, dim = -1, keepdim = True) + 1e-6
    mask = 1 - torch.eye(numAgents, device = device).unsqueeze(0)
    mask = mask.unsqueeze(-1)
    separation = ((difference / (distance**2)) * mask).sum(dim = 2)

    # Calculate Boundary Repulsion Force
    boundaryForce = calculateBoundaryRepulsion(positions, boundaryMargin, boundaryGain)

    # Combine Intended Actions & Agent/Wall Avoidance
    actions = action + separationGain * separation + boundaryForce

    # Clamp Actions to <= Unit Magnitude 1.0
    actions = torch.clamp(actions, -1.0, 1.0)
    return actions.transpose(0,1)

# Vectorised VMAS Port of original MPE 'simple_tag' controller - AI Assisted
def simpleTagController(env, boundaryMargin = 0.9, boundaryGain = 1.0):
    # Define Scenario Attributes & Separate Prey/Predators
    device = env.device
    numEnvs = env.num_envs
    numAgents = env.n_agents

    actions = torch.zeros(numEnvs, numAgents, 2, device=device)

    agents = env.world.agents

    predIndex = [i for i, a in enumerate(agents) if a.adversary]
    preyIndex = [i for i, a in enumerate(agents) if not a.adversary]

    predators = [agents[i] for i in predIndex]
    prey = [agents[i] for i in preyIndex]

    predPos = torch.stack([a.state.pos for a in predators], dim=1)  # (E,P,2)
    preyPos = torch.stack([p.state.pos for p in prey], dim=1)       # (E,R,2)

    # Predator Behaviour - Chase Nearest Prey
    difference = preyPos.unsqueeze(1) - predPos.unsqueeze(2)  # (E,P,R,2)
    distance = torch.norm(difference, dim=-1)                       # (E,P,R)

    nearestIdx = distance.argmin(dim=2)

    nearestPreyPos = torch.gather(
        preyPos,
        1,
        nearestIdx.unsqueeze(-1).expand(-1, -1, 2)
    )

    predatorAction = nearestPreyPos - predPos

    # Prey Behaviour - Avoid Nearest Predator
    difference = predPos.unsqueeze(1) - preyPos.unsqueeze(2)
    distance = torch.norm(difference, dim=-1)

    nearestIdx = distance.argmin(dim=2)

    nearestPredPos = torch.gather(
        predPos,
        1,
        nearestIdx.unsqueeze(-1).expand(-1, -1, 2)
    )

    preyAction = preyPos - nearestPredPos

    # Calculate Boundary Repulsion Force
    boundaryForce = calculateBoundaryRepulsion(preyPos, boundaryMargin, boundaryGain)
    preyAction = preyAction + boundaryForce

    # Create Actions for Each Agent - Uses Safe Indexing
    for k, idx in enumerate(predIndex):
        actions[:, idx] = predatorAction[:, k]

    for k, idx in enumerate(preyIndex):
        actions[:, idx] = preyAction[:, k]

    # Clamp Actions to <= Unit Magnitude 1.0
    norm = torch.norm(actions, dim=-1, keepdim=True) + 1e-6
    actions = actions / torch.clamp(norm, min=1.0)

    # Return Actions in Correct VMAS Format (Envs * Agents * 2)
    return actions.transpose(0, 1)

### RUNTIME

# Set Up Command Line Parsing - Short Tag
parser = argparse.ArgumentParser()
parser.add_argument('-st', '--shortTag', type = str, required = True)
parser.add_argument('-sc', '--scenario', type = str, required = True)
args = parser.parse_args()

# Define Simulation Hyperparameters
numEnvs = hyperparameters.NUM_ENVS
numAgents = hyperparameters.NUM_AGENTS
numSteps = hyperparameters.NUM_STEPS
scenario = args.scenario
scenarioParams = {}
rolesAgents = []
rolesLandmarks = []

# Hardcoded Index of Roles for Agents & Landmarks for Data Collection
# These are the Indexes for the Role One-Hot-Encoding
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

# Some Hyperparmeters Vary Between Scenarios
match(scenario):
        case "football":
            # Football Scenario has In-Built Controllers
            scenarioParams["ai_blue_agents"] = True
            scenarioParams["n_blue_agents"] = numAgents // 2
            scenarioParams["n_red_agents"] = numAgents // 2
            # Hardcoded Roles by Index
            rolesAgents = ["blueTeamAgent"] * (numAgents // 2) + ["redTeamAgent"] * (numAgents // 2)
            rolesLandmarks = ["physicsBall"]
        case "flocking":
            scenarioParams["n_agents"] = numAgents
            # Hardcoded Roles by Index
            rolesAgents = ["flockLeader"] + ["defaultAgent"] * numAgents
            rolesLandmarks = ["obstacle"] * 5 # Obstacle Number Could be Altered, Frozen at 5
        case "navigation":
            scenarioParams["n_agents"] = numAgents
            # Hardcoded Roles by Index
            rolesAgents = ["defaultAgent"] * numAgents
            rolesLandmarks = ["goalNavigation"] * numAgents
        case "transport":
            scenarioParams["n_agents"] = numAgents
            # Hardcoded Roles by Index
            rolesAgents = ["defaultAgent"] * numAgents
            rolesLandmarks = ["goalTransport", "physicsBox"]
        case "simple_tag": # 50/50 Split Predators & Prey
            scenarioParams["num_good_agents"] = numAgents // 2
            scenarioParams["num_adversaries"] = numAgents // 2
            # Hardcoded Roles by Index
            rolesAgents = ["tagPredator"] * (numAgents // 2) + ["tagPrey"] * (numAgents // 2)
            rolesLandmarks = ["obstacle"] * 2 # Obstacle Number Could be Altered, Frozen at 2

# Create a Roles List for Easy Lookup
roles = rolesAgents + rolesLandmarks

# Generate a Unqiue Run Seed
runSeed = np.random.randint(0, 2**31 - numEnvs)

# Create the Simulations - Changed to list to Allow per Sim RNG
env = vmas.make_env(
    scenario = scenario,
    num_envs = numEnvs,
    device = "cpu",
    continuous_actions = True,
    seed = runSeed,
    **scenarioParams
)

# Initialise/Reset Sims
env.reset()
describe_entities(env)
exit

print(f"GenerateData: Beginning {numEnvs} Parallel Simulations, {numAgents} Agents per Simulation, {numSteps} Timesteps Each")

# Create Empty Rows List
rows = []

# Iterate through Timesteps in All Parallel Simulations
for timestep in tqdm(range(numSteps), desc = "Simulation(s) Timesteps"):
    # OPTIONAL: Render Scenarios
    renderEnvGrid(env)

    # Pass Actions to Specific Scenario Controllers
    match(scenario):
        case "football":
            # Football Scenario has In-Built Controllers
            actions = env.get_random_actions()
        case "flocking":
            actions = flockingController(env)
        case "navigation":
            actions = navigationController(env)
        case "transport":
            actions = transportController(env)
        case "simple_tag":
            actions = simpleTagController(env)
    # Take a Step Forward in All Parallel Simulations
    env.step(actions)

    # Create List of all Tracked Entities in the Simulation
    trackedEntities = env.world.agents[:len(rolesAgents)] + env.world.landmarks[:len(rolesLandmarks)]

    # Enumerate through All Tracked Agents at that Timestep
    for entityId, entity in enumerate(trackedEntities):
        # Capture Postion & Velocities of the Particular Entity in Each Environment (Envs, 2)
        pos = entity.state.pos
        vel = entity.state.vel

        # Calculate Orientation from Velocity (Assumption) (Envs, 1)
        heading = torch.atan2(vel[:,1], vel[:,0])

        # Determine Role from the Lookup Table
        entityRole = roles[entityId]
        entityRoleIndex = roleDict[entityRole]

        # Create One-Hot-Encoding Feature Vector
        entityRoleOneHot = [0] * 11
        entityRoleOneHot[entityRoleIndex] = 1

        # Iterate through Individual Parallel Simulations
        for sceneId in range(numEnvs):
            # Create Row in .csv, Iterating through Timestep -> Agent -> Scene
            rows.append([
                sceneId,
                timestep,
                entityId,
                pos[sceneId, 0].item(),
                pos[sceneId, 1].item()
            ] + entityRoleOneHot + [
                vel[sceneId, 0].item(),
                vel[sceneId, 1].item(),
                heading[sceneId].item()
            ])

# Generate ID for New File & Create a New Folder
fileId = makeID("raw", args.shortTag)
folderPath = f"DataStore/RawData/{fileId}"
os.makedirs(folderPath, exist_ok=True)

# Write Recorded Data to a .csv File
with open(f"{folderPath}/{fileId}.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# Write Raw Data Metrics into a .json File
# Save the Model's Training Run Information - Losses, Model Attributes, Data Attributes
metrics = {
    "id": fileId,
    "seed": runSeed,
    "numEnvs": numEnvs,
    "numAgents": numAgents,
    "numSteps": numSteps,
    "scenario": scenario,
}
with open(f"{folderPath}/{fileId}.json", 'w') as file:
    json.dump(metrics, file, indent = 2)

print(f"GenerateData: Written Raw Simulation Data in {folderPath}")