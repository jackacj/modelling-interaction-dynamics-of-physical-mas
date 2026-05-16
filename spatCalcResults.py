import os
import json
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import hyperparameters
from ModelTemplates.extendedGNN import ExtendedGNN
from ModelTemplates.extendedMLP import ExtendedMLP
import matplotlib.pyplot as plt
import pandas as pd

"""
NOTED: THIS SCRIPT ASSUMES THAT THE LAST 15 MODELS IN 'DataStore/Models' ARE BEING USED IN THIS EXPERIMENT
"""

# Calculated ADE Values of a Model's Prediction & Ground Truth for Every KNearest Value in [0, 1, 2, 4, 8] - AI Assisted
def adePerHorizon(pred, gt):
    """
    pred, gt: [N, H, 2]
    returns: [N, H] ADE for each example at each horizon
    """
    displacement = torch.norm(pred - gt, dim=-1)   # [N, H]
    cumsum = torch.cumsum(displacement, dim=1)     # cumulative error
    steps = torch.arange(1, displacement.size(1)+1, device=displacement.device)
    ade = cumsum / steps                   # broadcast
    return ade

# Calculated FDE Values of a Model's Prediction & Ground Truth for Every KNearest Value in [0, 1, 2, 4, 8] - AI Assisted
def fdePerHorizon(pred, gt):
    """
    pred, gt: [N, H, 2]
    returns: [N, H] FDE at each horizon step
    """
    displacement = torch.norm(pred - gt, dim=-1)  # [N, H]
    return displacement

# Creates a Mask to select only Horizon Steps with Non-linear Movement, Serving NL-ADE Calculation - AI Assisted
def detectNonlinear(gt, angleThreshold=0.05):
    """
    gt: [N, H, 2]
    returns: mask [N, H] (bool) True where trajectory is nonlinear
    """

    vel = gt[:, 1:, :] - gt[:, :-1, :]
    velNorm = torch.nn.functional.normalize(vel, dim=-1)

    cosSimilarity = (velNorm[:, 1:, :] * velNorm[:, :-1, :]).sum(dim=-1)
    angleChange = torch.acos(torch.clamp(cosSimilarity, -1.0, 1.0))

    nonlinear = angleChange > angleThreshold  # [N, H-2]

    pad = torch.zeros(gt.size(0), 2, dtype=torch.bool, device=gt.device)
    nonlinearMask = torch.cat([pad, nonlinear], dim=1)

    return nonlinearMask

# Calculated NL-ADe Values of a Model's Prediction & Ground Truth for Every KNearest Value in [0, 1, 2, 4, 8] - AI Assisted
# Can Adjust the Angle Threshold for NL-ADE Mask
def nl_adePerHorizon(pred, gt, angleThreshold=0.05):
    """
    pred, gt: [N, H, 2]
    returns: [N, H] NL-ADE per horizon (NaN where undefined)
    """

    displacement = torch.norm(pred - gt, dim=-1)  # [N, H]
    nonlinearMask = detectNonlinear(gt, angleThreshold)

    maskedDisplacement = displacement * nonlinearMask

    cumsumDisplacement = torch.cumsum(maskedDisplacement, dim=1)
    cumsumCounts = torch.cumsum(nonlinearMask, dim=1)

    # Use NaN instead of 0 when no nonlinear steps exist
    nl_ade = cumsumDisplacement / cumsumCounts.clamp(min=1)

    # Explicitly set undefined positions to NaN
    nl_ade[cumsumCounts == 0] = float("nan")

    return nl_ade

# Calculated Means, Q1s & Q3s for each Error Metric for a Model's Prediction & Ground Truth - AI Assisted
def evaluateModel(model, dataloader, device):
    model.eval()

    all_ade = []
    all_fde = []
    all_nlade = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            gt = batch.y                # [N, H, 2]
            pred = model(batch)         # [N, H, 2]

            ade = adePerHorizon(pred, gt)
            fde = fdePerHorizon(pred, gt)
            nlade = nl_adePerHorizon(pred, gt)

            all_ade.append(ade.cpu())
            all_fde.append(fde.cpu())
            all_nlade.append(nlade.cpu())

    all_ade = torch.cat(all_ade, dim=0)
    all_fde = torch.cat(all_fde, dim=0)
    all_nlade = torch.cat(all_nlade, dim=0)

    def summarise(tensor):
        return (
            torch.nanmedian(tensor, dim=0).values.numpy(),
            torch.nanquantile(tensor, 0.25, dim=0).numpy(),
            torch.nanquantile(tensor, 0.75, dim=0).numpy(),
        )

    adeStats = summarise(all_ade)
    fdeStats = summarise(all_fde)
    nladeStats = summarise(all_nlade)

    return adeStats, fdeStats, nladeStats

##################################################################################################################################################################################################

batchSize = 1 # Calculating Errors for One Example at a Time
featureColSize = len(hyperparameters.FEATURE_COLUMNS)

models = {}

# Create List of Every Model: Scenario -> KNearest
modelDirs = [x[0] for x in os.walk("DataStore/Models")]

for modelDir in tqdm(modelDirs[-15:], desc = "Collecting Model Information"):
    # Get Each Model's Id & Tag
    modelId = modelDir[17:]
    modelTag = modelId.split('_')[-2]

    # Get Model Type, Source Dataset & Encoding Dim from Metrics JSON
    with open(f"{modelDir}/{modelId}.json", "r") as f:
        info = json.load(f)
        modelType = info["modelType"]
        sourceDataset = info["sourceDataset"]
        encodingDim = info["encoding"]
    
    # Get History, Horizon, KNearest & Raw Dataset from Source Dataset Metrics JSON
    with open(f"DataStore/Datasets/{sourceDataset}/{sourceDataset}.json", "r") as f:
        info = json.load(f)
        history = info["history"]
        horizon = info["horizon"]
        kNearest = info["kNearest"]
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
        model = ExtendedMLP(
            nodeInputDim = history * featureColSize,
            encodingDim = encodingDim,
            horizon = horizon
        )
    
    model.load_state_dict(torch.load(f'{modelDir}/{modelId}.pth', weights_only=True))

    # Create the Appropriate DataLoader for this Model's Test Set
    testDataSetPath = f"DataStore/Datasets/{sourceDataset}/{sourceDataset}_test.pt"
    testingDataset = torch.load(testDataSetPath) 
    testingLoader = DataLoader(testingDataset, batch_size = 1, shuffle = False)

    # Add Model & Info to Models Dictionary
    models[(scenario, kNearest, testingLoader)] = model

results = []

# Calculate Results
for (scenario, kNearest, loader), model in tqdm(models.items(), desc = "Calculating Model Results"):
    device = 'cpu'
    model = model.to(device)
    statsADE, statsFDE, statsNLADE = evaluateModel(model, loader, device)

    medADE, q1ADE, q3ADE = statsADE
    medFDE, q1FDE, q3FDE = statsFDE
    medNLADE, q1NLADE, q3NLADE = statsNLADE

    results.append({
            "scenario": scenario,
            "kNearest": kNearest,
            "medADE": float(medADE[-1]),
            "q1ADE": float(q1ADE[-1]),
            "q3ADE": float(q3ADE[-1]),
            "medFDE": float(medFDE[-1]),
            "q1FDE": float(q1FDE[-1]),
            "q3FDE": float(q3FDE[-1]),
            "medNLADE": float(medNLADE[-1]),
            "q1NLADE": float(q1NLADE[-1]),
            "q3NLADE": float(q3NLADE[-1]),
        })

##############################################################################################################################################################################################

metricToName = {
    "ADE": "Average Displacement Error",
    "FDE": "Final Displacement Error",
    "NLADE": "Non-Linear Average Displacement Error"
}

scenToName = {
    "football": "'football' Scenario (VMAS Simulation)",
    "simple_tag": "'simple_tag' Scenario (VMAS Simulation)",
    "little": "'little' Scenario (SDD Dataset)"
}

# AI Assisted Plotting Functions
def plotMetricForScenario(results, scenarioName, metricName):
    """
    metricName: 'ADE', 'FDE', or 'NLADE'
    """

    # Filter results for scenario
    scenario_data = [r for r in results if r["scenario"] == scenarioName]

    # Sort by K value
    scenario_data = sorted(scenario_data, key=lambda x: x["kNearest"])

    kValues = np.array([r["kNearest"] for r in scenario_data])
    medians = np.array([r[f"med{metricName}"] for r in scenario_data])
    q1 = np.array([r[f"q1{metricName}"] for r in scenario_data])
    q3 = np.array([r[f"q3{metricName}"] for r in scenario_data])

    # Asymmetric error bars (IQR)
    lowerError = medians - q1
    upperError = q3 - medians
    errors = [lowerError, upperError]

    x = np.arange(len(kValues))

    plt.figure(figsize=(8,5))
    plt.bar(x, medians, yerr=errors, capsize=5)

    plt.xticks(x, kValues)
    plt.xlabel("K Nearest Neighbours in Input Graph")
    plt.ylabel(f"{metricToName[metricName]} ({metricName})")
    plt.title(f"Connectivity against {metricName} - {scenToName[scenarioName]}")

    plt.tight_layout()
    plt.savefig(f"ExperimentalResults/Spatial/{metricName}_{scenario}.png")
    plt.show()

def plotAllMetricsForScenario(results, scenarioName):
    for metric in ["ADE", "FDE", "NLADE"]:
        plotMetricForScenario(results, scenarioName, metric)

scenarios = sorted(set(r["scenario"] for r in results))

for scenario in scenarios:
    plotAllMetricsForScenario(results, scenario)

# Save Results to a Results JSON
with open("ExperimentalResults/Spatial/results.json", "w") as resultsFile:
    json.dump(results, resultsFile, indent = 2)