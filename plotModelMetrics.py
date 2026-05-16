import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--modelId', type = str, required = True)
args = parser.parse_args()

# Open File Containing Model Metrics
with open(f"DataStore/Models/{args.modelId}/{args.modelId}.json") as file:
    metrics = json.load(file)

# Plot Training and Testing MSE Losses against Epoch
plt.plot(metrics["trainingLosses"], label="Train")
plt.plot(metrics["testingLosses"], label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()
