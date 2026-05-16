import subprocess

IdToShortTag = {
    "raw_20260228_1911_flockTemporal": "flock",
    "raw_20260228_1914_navTemporal" : "nav",
    "sdd_20260228_1916_nexusTemporal" : "nexus",
    "sdd_20260302_1115_coupaTemporal" : "coupa"
}

horizon = 20
kNearest = 2
rawId = "sdd_20260302_1115_coupaTemporal"

# for rawId in ["raw_20260228_1911_flockTemporal", "raw_20260228_1914_navTemporal", "sdd_20260228_1916_nexusTemporal"]:
for history in [1, 3, 5, 10, 20]:
    subprocess.run([
        "projectVenv310\Scripts\python.exe", 
        ".\processData.py", 
        "-i", rawId, 
        "-st", f"{IdToShortTag[rawId]}h{history}t{horizon}", 
        "-hy", str(history), 
        "-hn", str(horizon),
        "-k", str(kNearest)
        ])