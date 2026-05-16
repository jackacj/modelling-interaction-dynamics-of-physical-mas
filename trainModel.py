import torch
from torch_geometric.loader import DataLoader
import os
import json 
import hyperparameters
import argparse
from datetime import datetime
from tqdm import tqdm

def makeID(stage, tag):
    now = datetime.now()
    return f"{stage}_{now:%Y%m%d_%H%M}_{tag}"

# Import Different Model Types
from ModelTemplates.extendedGNN import ExtendedGNN
from ModelTemplates.baselineGNN import BaselineGNN
from ModelTemplates.extendedMLP import ExtendedMLP
from ModelTemplates.baselineMLP import BaselineMLP

# Set Up Command Line Parsing - Input Path for Training & Testing Sets (Id) - Model Type - Short Tag
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataId', type = str, required = True)
parser.add_argument('-st', '--shortTag', type = str, required = True)
parser.add_argument('-m', '--modelType', type = str, required = True) # EG, EM, BG, BM - Different Types of Models (E - Extended, B - Baseline, G - GNN, M - MLP)
args = parser.parse_args()

# Read History & Horizon from Dataset
with open(f"DataStore/Datasets/{args.dataId}/{args.dataId}.json", "r") as f:
    data = json.load(f)
    history = data["history"]
    horizon = data["horizon"]

# Define Training Hyperparemeters (Temporary Values)
numEpochs = hyperparameters.NUM_EPOCHS
batchSize = hyperparameters.BATCH_SIZE
trainingRatio = hyperparameters.TRAINING_RATIO
encodingDim = hyperparameters.ENCODING_DIM
featureColumnLength = len(hyperparameters.FEATURE_COLUMNS)
dataColumns = hyperparameters.DATA_COLUMNS

# Load Training & Testing Sets from Datasets
trainDataSetPath = f"DataStore/Datasets/{args.dataId}/{args.dataId}_train.pt"
testDataSetPath = f"DataStore/Datasets/{args.dataId}/{args.dataId}_test.pt"

trainingDataset = torch.load(trainDataSetPath) 
testingDataset = torch.load(testDataSetPath) 

trainingLoader = DataLoader(trainingDataset, batch_size = batchSize, shuffle = True)
testingLoader = DataLoader(testingDataset, batch_size = batchSize, shuffle = False)

# Instantiate Model of Given Type
match args.modelType:
    case "EG":
        model = ExtendedGNN(
            nodeInputDim = history * featureColumnLength,
            edgeInputDim = 3,
            encodingDim = encodingDim,
            horizon = horizon,
        )
    case "EM":
        model = ExtendedMLP(
            nodeInputDim = history * featureColumnLength,
            encodingDim = encodingDim,
            horizon = horizon,
        )
    case "BG":
        model = BaselineGNN(
            nodeInputDim = history * featureColumnLength,
            edgeInputDim = 3,
            encodingDim = encodingDim,
            horizon = horizon,
        )
    case "BM":
        model = BaselineMLP(
            nodeInputDim = history * featureColumnLength,
            horizon = horizon,
        )
    case _:
        raise Exception("Argument --modelType expects any one of these inputs: EG, EM, BG, BM")

# Create an Optimiser (Fine Tune Learning Rate when Testing)
optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)

print(f"Training Script: Commencing Training with {numEpochs} Epochs and Batch Size {batchSize}")

# Generate ID for New File & Create a New Folder
fileId = makeID("mdl", args.shortTag)
folderPath = f"DataStore/Models/{fileId}"
os.makedirs(folderPath, exist_ok=True)

modelFilePath = f"{folderPath}/{fileId}.pth"
metricsFilePath = f"{folderPath}/{fileId}.json"

# Prepare to Collect Epoch Training & Testing Losses for Future Use
trainingLosses = []
testingLosses = []
bestTestLoss = float("inf")
bestEpoch = -1

# Define Epoch & Training/Testing Loops
for epoch in tqdm(range(numEpochs), desc = "Training Epochs"):
    model.train()
    trainingLoss = 0
    # Define Training Loop for a Batch
    for batch in trainingLoader:
        # Reset Gradients, Generate Predictions & Calculate Loss
        optimiser.zero_grad()
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction, batch.y)

        # Optimise Model according to Loss
        loss.backward()
        optimiser.step()

        trainingLoss += loss.item()
    
    # Average Training Loss by Loader Size
    trainingLoss /= len(trainingLoader)

    # Set Up 'Evaluation while Training'
    model.eval()
    testingLoss = 0
    # Disable Gradient Calculation in Testing Block (Speed Up Testing)
    with torch.no_grad():
        for batch in testingLoader:
            # Generate Predictions & Calculate Loss
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y)

            testingLoss += loss.item()
    
    # Average Testing Loss by Loader Size
    testingLoss /= len(testingLoader)

    # Display Epoch Number, Training Loss & Testing Loss
    print(f"Epoch {epoch}, Training Loss {trainingLoss}, Testing Loss {testingLoss}")

    # Check for Best Test Loss, Save Epoch Model if So
    if testingLoss < bestTestLoss:
        bestTestLoss = testingLoss
        bestEpoch = epoch

        # Save the Model 
        torch.save(model.state_dict(), modelFilePath)

    # Save Training & Testing Losses for this Epoch
    trainingLosses.append(trainingLoss)
    testingLosses.append(testingLoss)

# Save the Model's Information
metricsInfo = {
    "id": fileId,
    "sourceDataset": args.dataId,
    "encoding": encodingDim,
    "epochs": numEpochs,
    "bestEpoch": bestEpoch,
    "batchSize": batchSize,
    "modelType": args.modelType,
    "trainingLosses": trainingLosses,
    "testingLosses": testingLosses,
}

with open(metricsFilePath, 'w') as file:
    json.dump(metricsInfo, file, indent = 2)