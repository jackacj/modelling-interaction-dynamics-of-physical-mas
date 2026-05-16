import os
import csv
import json
import math
import argparse
from tqdm import tqdm
from datetime import datetime

# ID Helper Function
def makeID(stage, tag):
    now = datetime.now()
    return f"{stage}_{now:%Y%m%d_%H%M}_{tag}"

# Pass Command Line Arguments (Annotations File Name & Short Tag)
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--sceneFolder', type = str, required = True)
parser.add_argument('-t', '--timePerTimestep', type = float, required = True)
parser.add_argument('-st', '--shortTag', type = str, required = True)
args = parser.parse_args()

# Set Data Sampling Values
fps = 30 # Static for All SDD Videos
timePerTimestep = args.timePerTimestep # Measured in Seconds
chunkSize = int(fps * timePerTimestep) # Number of Frames Aggregated Together

# Hardcoded Number of Videos per Scene - Can Add More
videoCount = {
    "nexus": 12,
    "quad": 4,
    "coupa": 4,
    "little": 4
}

# Construct Path to Read SDD Data
folderPath = f"DataStore/SDD/{args.sceneFolder}"
filePaths = [f"{folderPath}/video{i}.txt" for i in range(videoCount[args.sceneFolder])]

# Construct Role Mapper
roleMap = {
    "Pedestrian": 0,
    "Biker": 1,
    "Skater": 2,
    "Car": 3,
    "Bus": 4,
    "Cart": 5
}

# Create Empty Rows List
rows = []

# Loop through Every Annotations File for a Given Scene
for sceneId, filePath in tqdm(enumerate(filePaths), desc = f"SDD '{args.sceneFolder}' Annotation Files"):
    # Track Agent Data for Future Aggregation - AgentId/TrackId -> [{frame, posX, posY, role}]
    agentData = {}

    # Open the Annotations File & Read Row Data
    with open(filePath, "r") as sddFile:
        # Read Each Line
        for line in sddFile:
            # Extract Information from Line
            parts = line.strip().split()

            trackId = int(parts[0]) # Individual Track (Matches onto AgentId)
            xmin = float(parts[1]) # Bounding Box (Can Construct Positions)
            ymin = float(parts[2]) #
            xmax = float(parts[3]) #
            ymax = float(parts[4]) #
            frame = int(parts[5]) # Timestep
            lost = int(parts[6]) # Lost - Agent is Gone/Ended Track - Remove these Lines
            occluded = int(parts[7]) # Occluded - Less Accurate
            generated = int(parts[8]) # Linearly Interpolated - If 0: Hand Annotated
            label = parts[9].strip('"') # Textual Role

            # Filter Out Rows which are Lost (Generated & Occluded are fine)
            if (lost == 1):
                # Skip this Row
                continue

            # Calculate Position (Agent at Timestep)
            posX = (xmax + xmin) // 2
            posY = (ymax + ymin) // 2

            # Collect Data for Future Aggregation (Agent at Timestep)
            if trackId not in agentData:
                agentData[trackId] = []
            agentData[trackId].append({
                "frame": frame,
                "posX": posX,
                "posY": posY,
                "role": label
            })

    # Store All Agent Chunks - (AgentId, Chunk Data (Range of Frames))
    allChunks = []
    globalStartFrames = set()

    # Iterate through All Agents in Scene for Chunking
    # 1st Pass: Build Chunks & Collect Global Start Frames
    for trackId, frames in agentData.items():
        # Sort Agent Frames in Order
        frames.sort(key = lambda x : x["frame"])

        # Iterate through Every Agent Chunk
        for i in range(0, len(frames), chunkSize):
            chunk = frames[i: i + chunkSize]

            # Skip Empty Chunks
            if len(chunk) == 0:
                continue

            # Add Start Frame for this Chunk & Save Chunk
            startFrame = chunk[0]["frame"]
            globalStartFrames.add(startFrame)
            allChunks.append((trackId, chunk))

    # Build Contiguous Timestep Mapping
    sortedFrames = sorted(globalStartFrames)
    frameToStep = {frame: i for i, frame in enumerate(sortedFrames)}
    
    # Create Nested Dict to Hold Agent Timelines - {AgentId -> *{Timestep -> [Row Data]}}
    agentTimelines = {}

    # 2nd Pass: Compute Agent Features & Save Rows
    for trackId, chunk in allChunks:
        # Get Scene-Wide Consistent Timestep
        startFrame = chunk[0]["frame"]
        timestep = frameToStep[startFrame]
            
        # Calculate Change in Time - Difference Between Last & First Frame in Chunk & Turned to Real Time
        dt = (chunk[-1]["frame"] - chunk[0]["frame"]) / fps

        # Avoid Stagnant/Backwards Time Flow
        if dt <= 0:
            dt = 1 / fps

        # Find Mean Position across Chunk
        meanX = sum(fr["posX"] for fr in chunk) / len(chunk)
        meanY = sum(fr["posY"] for fr in chunk) / len(chunk)

        # Calculate Velocity over Chunk - From dx & dt
        velX = (chunk[-1]["posX"] - chunk[0]["posX"]) / dt
        velY = (chunk[-1]["posY"] - chunk[0]["posY"]) / dt

        # Calculate Heading Based on Velocity
        heading = math.atan2(velY, velX) if (velX != 0 or velY != 0) else 0.0

        # Calculate Role One-Hot-Encoding
        # There are 6 Roles in SDD but we use a Redudant 11 Role Long Encoding to Fit into Existing Data Infrastructure
        # Agent's have Consistent Roles (Don't Change over Time)
        roleVector = [0] * 11
        roleId = roleMap[chunk[0]["role"]]
        roleVector[roleId] = 1

        # Construct Output Row
        row = [
            sceneId, # SceneId
            timestep, # Timestep - Now Scaled to Real Time Units
            trackId, # AgentId
            meanX, # PosX
            meanY, # PosY
            *roleVector, # Role One-Hot-Encoding Vector (agentRole{0 -> 10})
            velX, # VelX
            velY, # VelY
            heading, # Heading
        ]

        # Store Agent's Row Data for a Given Timestep
        if trackId not in agentTimelines:
            agentTimelines[trackId] = {}
        agentTimelines[trackId][timestep] = row

    # 3rd Pass: Fill Missing Timesteps for Every Agent
    for trackId, timeline in agentTimelines.items():
        # Get the Existing Timesteps for Agent
        existingSteps = sorted(timeline.keys())

        # Find the Earliest & Latest Steps
        minStep = existingSteps[0]
        maxStep = existingSteps[-1]

        # Iterate through Range between Earliest & Latest Steps
        lastRow = None
        for step in range(minStep, maxStep + 1):
            # Step Already Exists -> Save Row Data as Last Row, Add Row
            if step in timeline:
                lastRow = timeline[step]
                rows.append(lastRow)
            # Step Doesn't Exist -> Carry Forward Last Known Row, Add Row
            else:
                if lastRow is None:
                    continue
                filledRow = lastRow.copy()
                filledRow[1] = step
                rows.append(filledRow)

# Write Output to .csv
# Generate ID for New File & Create a New Folder
fileId = makeID("sdd", args.shortTag)
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
    "scenario": args.sceneFolder,
    "numVideos": videoCount[args.sceneFolder],
    "timePerTimestep": args.timePerTimestep,
}
with open(f"{folderPath}/{fileId}.json", 'w') as file:
    json.dump(metrics, file, indent = 2)

print(f"ConvertSDDtoCSV: Converted SDD Annotations into Raw Data in {folderPath}")
