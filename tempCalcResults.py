import os
import json
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import hyperparameters
from ModelTemplates.extendedGNN import ExtendedGNN
from ModelTemplates.baselineGNN import BaselineGNN
import matplotlib.pyplot as plt
import pandas as pd

"""
NOTED: THIS SCRIPT ASSUMES THAT ALL MODEL DIRS IN 'DataStore/Models/' ARE BEING USED IN THIS EXPERIMENT
"""

# Calculated ADE Values of a Model's Prediction & Ground Truth for Every Horizon Value 1 -> 20 - AI Assisted
def adePerHorizon(pred, gt):
    """
    pred, gt: [N, H, 2]
    returns: [N, H] ADE for each example at each horizon
    """
    disp = torch.norm(pred - gt, dim=-1)   # [N, H]
    cumsum = torch.cumsum(disp, dim=1)     # cumulative error
    steps = torch.arange(1, disp.size(1)+1, device=disp.device)
    ade = cumsum / steps                   # broadcast
    return ade

# Calculated Mean ADE, Q1, Q3 Values of a Model's Prediction & Ground Truth - AI Assisted
def evaluateModel(model, dataloader, device):
    model.eval()
    all_ade = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            gt = batch.y                # [N, H, 2]
            pred = model(batch)         # [N, H, 2]

            ade = adePerHorizon(pred, gt)  # [N, H]

            all_ade.append(ade.cpu())

    all_ade = torch.cat(all_ade, dim=0)    # [N_total, H]

    mean = all_ade.mean(dim=0).numpy()
    q1   = torch.quantile(all_ade, 0.25, dim=0).numpy()
    q3   = torch.quantile(all_ade, 0.75, dim=0).numpy()

    return mean, q1, q3

#################################################################################################################################################################################

batchSize = hyperparameters.BATCH_SIZE
featureColSize = len(hyperparameters.FEATURE_COLUMNS)

models = {}

# Create List of Every Model: Model Type -> Scenario -> Burn-In
modelDirs = [x[0] for x in os.walk("DataStore/Models")]

for modelDir in tqdm(modelDirs[1:], desc = "Collecting Model Information"):
    # Get Each Model's Id & Tag
    modelId = modelDir[17:]
    modelTag = modelId.split('_')[-1]

    # Get Model Type, Source Dataset & Encoding Dim from Metrics JSON
    with open(f"{modelDir}/{modelId}.json", "r") as f:
        info = json.load(f)
        modelType = info["modelType"]
        sourceDataset = info["sourceDataset"]
        encodingDim = info["encoding"]
    
    # Get History, Horizon & Raw Dataset from Source Dataset Metrics JSON
    with open(f"DataStore/Datasets/{sourceDataset}/{sourceDataset}.json", "r") as f:
        info = json.load(f)
        history = info["history"]
        horizon = info["horizon"]
        rawDataset = info["sourceRawData"]

    # Get Scenario from Raw Data Dataset Metrics JSON
    with open(f"DataStore/RawData/{rawDataset}/{rawDataset}.json", "r") as f:
        info = json.load(f)
        scenario = info["scenario"]
    
    # Load the Model
    if modelType == "EG":
        model = ExtendedGNN(
            nodeInputDim = history * featureColSize,
            edgeInputDim = 3,
            encodingDim = encodingDim,
            horizon = horizon
        )
    else:
        model = BaselineGNN(
            nodeInputDim = history * featureColSize,
            edgeInputDim = 3,
            encodingDim = encodingDim,
            horizon = horizon
        )

    model.load_state_dict(torch.load(f'{modelDir}/{modelId}.pth', weights_only=True))

    # Create the Appropriate DataLoader for this Model's Test Set
    testDataSetPath = f"DataStore/Datasets/{sourceDataset}/{sourceDataset}_test.pt"
    testingDataset = torch.load(testDataSetPath) 
    testingLoader = DataLoader(testingDataset, batch_size = 1, shuffle = False)

    # Add Model & Info to Models Dictionary
    models[(modelType, scenario, history, testingLoader)] = model

results = []

# Calculate Results
for (modelType, scenario, history, loader), model in tqdm(models.items(), desc = "Calculating Model Results"):
    device = 'cpu'
    model = model.to(device)
    mean, q1, q3 = evaluateModel(model, loader, device)

    horizonLength = len(mean)
    for horizon in range(horizonLength):
        results.append({
            "modelType": modelType,
            "scenario": scenario,
            "history": history,
            "horizon": horizon+1,
            "meanADE": float(mean[horizon]),
            "q1": float(q1[horizon]),
            "q3": float(q3[horizon])
        })

# Convert Results to Dataframe
df = pd.DataFrame(results)

typeToName = {
    "EG": "Model w/ Message Passing + Temporal Encoder",
    "BG": "Model w/ Message Passing Only"
}

scenToName = {
    "flocking": "'flocking' Scenario (VMAS Simulation)",
    "navigation": "'navigation' Scenario (VMAS Simulation)",
    "coupa": "'coupa' Scenario (SDD Dataset)"
}

# Create a Plot for Each Combination of Model Type & Scenario
# Y Axis is ADE, X Axis is the Horizon
# Each Shaded Line on a Graph is a Model trained on a different History
for (modelType, scenario), subDf in df.groupby(["modelType", "scenario"]):
    plt.figure()

    for history, axisDf in subDf.groupby("history"):
        axisDf = axisDf.sort_values("horizon")

        plt.plot(axisDf["horizon"], axisDf["meanADE"], label = f"History = {history}")
        plt.fill_between(
            axisDf["horizon"],
            axisDf["q1"],
            axisDf["q3"],
            alpha = 0.2
        )

    plt.title(f"{typeToName[modelType]} – {scenToName[scenario]}")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("Average Displacement Error (ADE)")
    plt.legend()
    plt.savefig(f"ExperimentalResults/Temporal/{modelType}_{scenario}.png")
    plt.show()

# Save Results to a Results JSON
with open("ExperimentalResults/Temporal/results.json", "w") as resultsFile:
    json.dump(results, resultsFile, indent = 2)