import subprocess

IdToShortTag = {
    "raw_20260303_1329_footballSpatial": "football",
    "raw_20260303_1329_tagSpatial": "tag",
    "sdd_20260303_1331_littleSpatial": "little"
}

horizon = 10
history = 5

for rawId in ["raw_20260303_1329_footballSpatial", "raw_20260303_1329_tagSpatial", "sdd_20260303_1331_littleSpatial"]:
    for kNearest in [0, 1, 2, 4, 8]:
        subprocess.run([        
            "projectVenv310\Scripts\python.exe", 
            ".\processData.py", 
            "-i", rawId, 
            "-st", f"{IdToShortTag[rawId]}_k{kNearest}", 
            "-hy", str(history), 
            "-hn", str(horizon),
            "-k", str(kNearest)
        ])