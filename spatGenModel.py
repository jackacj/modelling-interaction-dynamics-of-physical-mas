import subprocess
import os
import json

directories = [x[0] for x in os.walk("DataStore/Datasets")]

for directory in directories[-15:]:
    dataId = directory[19:]
    files = os.listdir(directory)

    metrics = files[0]
    testSet = files[1]
    trainSet = files[2]

    # Get Info from Metrics
    with open(f"{directory}/{metrics}", "r") as f:
        info = json.load(f)
        history = info["history"]
        horizon = info["horizon"]
        kNearest = info["kNearest"]

    baseTag = dataId.split('_')[-2]

    if (kNearest == 0):
        modelTag = "EM"
    else:
        modelTag = "EG"

    subprocess.run([
        "projectVenv310\Scripts\python.exe", 
        ".\\trainModel.py", 
        "-i", dataId, 
        "-st", f"{baseTag}_k{kNearest}_REPLACE", 
        "-m", modelTag 
    ])