import subprocess
import os
import json

directories = [x[0] for x in os.walk("DataStore/Datasets")]

# Only Do for Coupa
for directory in directories[-5:]:
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
        sourceId = info["sourceRawData"]
    
    baseTag = sourceId.split('_')[-1]

    subprocess.run([
        "projectVenv310\Scripts\python.exe", 
        ".\\trainModel.py", 
        "-i", dataId, 
        "-st", f"{baseTag}h{history}t{horizon}", 
        "-m", "EG" 
    ])

    subprocess.run([
        "projectVenv310\Scripts\python.exe", 
        ".\\trainModel.py", 
        "-i", dataId, 
        "-st", f"{baseTag}h{history}t{horizon}", 
        "-m", "BG" 
    ])
