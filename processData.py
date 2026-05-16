import numpy as np
import pandas as pd
import os
import json
import torch
import torch_cluster
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from tqdm import tqdm
import hyperparameters
import argparse
from datetime import datetime

def makeID(stage, tag):
    now = datetime.now()
    return f"{stage}_{now:%Y%m%d_%H%M}_{tag}"

### CONVERTING RAW .CSV DATA -> TRAINING EXAMPLE DICTS

# Convert Scene Dataframe into a Sorted List of Timesteps & [Timestep][Agent] Dictionary for Features
def sceneToDict(sceneDf):
    # Create the Feature Dictionary indexed by the Timestep & AgentID
    data = {}
    for (t, a), row in sceneDf.set_index(["timestep", "agentId"]).iterrows():
        data.setdefault(t, {})[a] = row
    
    # Return Sorted Timesteps & Dictionary
    timesteps = sorted(data.keys())
    return timesteps, data

# Check if an Agent in a Scene is Continuous (Doesn't Disappear)
def isAgentContinuous(agentId, t0, data, history, horizon):
    # Generate Required Timesteps
    requiredTimesteps = list(range(t0 - history + 1, t0 + horizon + 1))
    for t in requiredTimesteps:
        if t not in data:
            # Return False if there is Not Every Required Timestep
            return False
        if agentId not in data[t]:
            # Return False if the Agent is Not Present in a Given Timestep
            return False
    
    # Else, Return True
    return True

# Extract a Training Example from an Anchor Timestep
def extractAtAnchor(t0, data, history, horizon):
    # Extract List of Agents, Filtering for Continuity
    agentsAtAnchor = set(data[t0].keys())
    agents = [agent for agent in agentsAtAnchor if isAgentContinuous(agent, t0, data, history, horizon)]

    # Return No Example if Too Many Agents are Filtered
    if len(agents) < minAgents:
        return None
    
    # Consistently Order Agents & Determine Training Example Properties
    agents = sorted(agents)
    count = len(agents)
    featureDim = len(featureColumns)

    # Build the "nodeHistory" Tensor
    nodeHistory = np.zeros((count, history, featureDim), dtype = np.float32)
    for index, agent in enumerate(agents):
        for jndex, timestep in enumerate(range(t0 - history + 1, t0 + 1)):
            # Enumerate through Agents & Timesteps, Adding Feature Rows into "nodeHistory"
            row = data[timestep][agent]
            nodeHistory[index, jndex] = row[featureColumns].values

    # Build the "pos" Table
    pos = np.zeros((count, 2), dtype = np.float32)
    for index, agent in enumerate(agents):
        # Enumerate through Agents, Populating "pos" with Agent Positions
        row = data[t0][agent]
        pos[index] = [row["posX"], row["posY"]]
    
    # Build the "futurePos" Tensor - Using Relative Displacements
    futurePos = np.zeros((count, horizon, 2), dtype = np.float32)
    for index, agent in enumerate(agents):
        # Enumerate through Agents, Extracting Anchor Position
        x0, y0 = data[t0][agent][["posX", "posY"]].values
        for jndex, timestep in enumerate(range(t0 + 1, t0 + horizon + 1)):
            # Enumerate through Agents & Timesteps, Populating with Relative Displacements
            row = data[timestep][agent]
            futurePos[index, jndex] = [row["posX"] - x0, row["posY"] - y0]

    # Return Training Example
    return {
        "nodeHistory": nodeHistory,
        "pos": pos,
        "futurePos": futurePos
    }

# Extract Multiple Examples from a Scene using a Sliding Window
def extractFromScene(sceneDf, history, horizon):
    # Convert Scene Dataframe into a Timestep List & Data Dictionary
    timesteps, data = sceneToDict(sceneDf)

    # Define Bounds
    examples = []
    minT = min(timesteps)
    maxT = max(timesteps)

    # Iterate through All Windows (Anchored at "t0")
    for t0 in range(minT + history - 1, maxT - horizon + 1):
        # If Timestep Doesn't Exist, Skip to Next
        if t0 not in data:
            continue
        
        # Extract a Training Example at Anchor
        example = extractAtAnchor(t0, data, history, horizon)
        if example is None:
            continue

        # Filter for Corrupted Examples (None Values)
        requiredKeys = ["nodeHistory", "pos", "futurePos"]
        if not all(key in example for key in requiredKeys):
            continue

        if any(example[key] is None for key in requiredKeys):
            continue

        # Add Safe Example to List
        examples.append(example)

    # Return List of Extracted Examples
    return examples

# Extract an Example Dataset from an Entire Log File
def extractFromLog(logDf, history, horizon, trainingRatio):
    trainingDataset = []
    testingDataset = []

    # Seperate Out Scenes in Log Dataframe
    scenes = list(logDf.groupby("sceneId"))

    # Split Training & Testing Example Ids Deterministically
    torch.manual_seed(0)

    for sceneId, sceneDf in tqdm(scenes, desc = "Data Scenes"):
        sceneExamples = extractFromScene(sceneDf, history, horizon)
        if sceneExamples:
            splitIndex = int(trainingRatio * len(sceneExamples))
            trainingDataset.extend(sceneExamples[:splitIndex])
            testingDataset.extend(sceneExamples[splitIndex:])
    
    # Return Dataset
    return trainingDataset, testingDataset

### CONVERTING TRAINING EXAMPLE DICTS -> PYG DATA OBJECTS

# Build Edge Index & Features from Agent Positions (Nodes, 2) using K-NN
def buildEdges(pos, k):
    # Create Index of Shape (2, Nodes*Neighbours)
    edge_index = knn_graph(pos, k = k, loop = False)

    # Calculate Relative Position (Edges, 2) & Distance (Edges, 1) from Source Node (Row) to Target Node (Col)
    row, col = edge_index
    relativePos = pos[col] - pos[row]
    distance = relativePos.norm(dim = 1, keepdim = True)

    # Create Edge Features (Edges, 3) - Relative Position .. Distance
    edge_attr = torch.cat([relativePos, distance], dim = 1)
    return edge_index, edge_attr

def exampleToDataObject(sample, k):
    # Extract Fields from Test Case & Convert NumPy -> Torch Tensors
    nodeHistory = torch.from_numpy(sample["nodeHistory"]).float()
    pos = torch.from_numpy(sample["pos"]).float()
    futurePos = torch.from_numpy(sample["futurePos"]).float()

    # Flatten Node History (Nodes, Timesteps, Features) -> (Nodes, Timesteps*Features)
    x = nodeHistory.view(nodeHistory.size(0), -1)

    # Build Edge Index & Features (Temporary K Value)
    edge_index, edge_attr = buildEdges(pos, k)
        
    # Return PyG Graph Object
    return Data(
        x = x,
        edge_index = edge_index,
        edge_attr = edge_attr,
        y = futurePos,
    )

### RUNTIME EXECUTION

print(f"ProcessData: Importing Hyperparameters")

# Set Up Command Line Parsing - Input for Raw Data (Id) - Short Tag
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--rawId', type = str, required = True)
parser.add_argument('-st', '--shortTag', type = str, required = True)
parser.add_argument('-hy', '--history', type = int, required = True)
parser.add_argument('-hn', '--horizon', type = int, required = True)
parser.add_argument('-k', '--kNearest', type = int, required = True)
args = parser.parse_args()

# Define Hyperparameters (Temporary Values)
minAgents = hyperparameters.AGENT_MIN
featureColumns = hyperparameters.FEATURE_COLUMNS
dataColumns = hyperparameters.DATA_COLUMNS
history = args.history
horizon = args.horizon
trainingRatio = hyperparameters.TRAINING_RATIO
k = args.kNearest

# Define Raw Data Filepath
rawFilePath = f"DataStore/RawData/{args.rawId}"

# Open the Raw Data File and Add Column Names
df = pd.read_csv(f"{rawFilePath}/{args.rawId}.csv", header=None)
df.columns = dataColumns

print(f"ProcessData: Extracting Training & Testing Examples from {rawFilePath}")

# Extract Examples from File & Creating Training & Testing Datasets (Lists)
trainingExamples, testingExamples = extractFromLog(df, history, horizon, trainingRatio)

trainingDataList = [exampleToDataObject(s, k) for s in trainingExamples]
testingDataList = [exampleToDataObject(s, k) for s in testingExamples]

# Generate ID for New File & Create a New Folder
fileId = makeID("ds", args.shortTag)
folderPath = f"DataStore/Datasets/{fileId}"
os.makedirs(folderPath, exist_ok=True)

# Save Training & Testing Datasets
trainSetFilePath = f"{folderPath}/{fileId}_train.pt"
testSetFilePath = f"{folderPath}/{fileId}_test.pt"

torch.save(trainingDataList, trainSetFilePath)
torch.save(testingDataList, testSetFilePath)

# Write Raw Data Metrics into a .json File
# Save the Model's Training Run Information - Losses, Model Attributes, Data Attributes
metrics = {
    "id": fileId,
    "sourceRawData": args.rawId,
    "history": history,
    "horizon": horizon,
    "kNearest": k,
    "minAgents": minAgents,
    "trainingRatio": trainingRatio,
    "numExamples": len(trainingExamples) + len(testingExamples),
    "numTraining": len(trainingExamples),
    "numTesting": len(testingExamples)
}
with open(f"{folderPath}/{fileId}.json", 'w') as file:
    json.dump(metrics, file, indent = 2)

print(f"ProcessData: Saved Training & Testing Data to {folderPath}")