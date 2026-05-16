### Project Overview

This project is an investigation into behaviour prediction in multi-agent systems. This repository contains every different facet of this project including research, designs, implementations and sample data. The core of this project is a simulation library paired with custom defined PyTorch Geometric (PyG) models which use processed simulation data and supervised learning to perform multi-agent behaviour prediction. The goal is to investigate the challenges of this task through evaluating design features of a custom approach informed by research through controlled experimentation.

### Technical Setup

This project was programmed in **Python 3.10** running on **Windows 10**

Run the following commands in the root directory to create the python virtual environment required to run the simulation, this uses Python 3.10 as later versions cause issues with the stability of specific libraries:

```shell
py -3.10 -m venv venv310;
source venv/Scripts/activate;
pip install -r requirements.txt;
```

Once this is created and activated you will be able to interface with the project.

### Repository Layout

##### Interface Scripts

The repository is formatted so all interfaces into the pipeline are easily accessible via the terminal in the root of the directory.

> Note: Various scripts throughout the project draw certain variables from `hyperparameters.py`. These act as global settings across the project, you should make sure this file is always set to the values you intend before executing any part of this project.

Below is a list of the scripts accessible from the root which act as interfaces into the pipeline.

- `generateData.py`: This script governs the instantiation, execution and data collection from parallel multi-agent simulations. Leverages the library 'VMAS' (https://vmas.readthedocs.io/en/stable/index.html). Takes the arguments "Short Tag" `-st` and "Scenario" `-sc`. This generates a new directory `DataStore/RawData/raw_#` and creates a raw data file and an information file in the forms `raw_#.csv` and `raw_#.json`.

- `convertSDDtoCSV.py`: This script converts locally stored annotations text files taken from the 'Stanford Drone Dataset' (https://cvgl.stanford.edu/projects/uav_data/) into the raw data schema used by the rest of this pipeline. Takes the arguments "Short Tag" `-st`, "Scene Folder" `-i`, and "Time per Timestep" `-t`. This generates a new directory `DataStore/RawData/sdd_#` and creates a raw data file and an information file in the forms `sdd_#.csv` and `sdd_#.json`.

- `processData.py`: This script processes raw data files into PyG compatible training & testing datasets. Takes the arguments "Short Tag" `-st`, "Input" `-i` in the form `{raw || sdd}_#`, "History" `-h`, "Horizon" `-h`, and "K-Nearest" `-k`. This generates a new directory `DataStore/Datasets/ds_#` and creates two dataset files and an information file in the forms `ds_#_train.pt`, `ds_#_test.pt` and `ds_#.json`.

- trainModel.py: This script trains an instance of a given type of model with a training & testing dataset and returns a saved model as well as information about the model's performance and attributes. Takes the arguments "Short Tag" `-st`, "Input" `-i` in the form `ds_#`, and "Model Type" `-m` in the form `{EG || EM || BG || BM}`. This generates a new directory `DataStore/Models/mdl_#` and creates a model weights file and an information file in the forms `mdl_#.pth` and `mdl_#.json`.

In the root directory there is also `tempCalcResults.py` & `spatCalcResults.py`. These are used to calculate/plot the experimental results of the **Temporal Modelling** & **Interaction Modelling** experiments respectively (detailed in the project report) but these scripts are hardcoded with inputs and make assertions about how the rest of the repository is structured so are unlikely to work or produce meaningful output without alteration if cloning the repository.

##### Data Store & Directories

All the data used within different parts of the pipeline is stored within a directory in the `DataStore` directory. For each entry within these directories (excluding `SDD` & `Demos`) an information file is included giving an overview of the specific conditions the data was produced within, this is stored as a `.json`. Below are the directories within `DataStore`:

- `Datasets`: Stores processed data in the form of PyG `Data` objects, stored as `.pt` files and used as input for training models.

- `Models`: Trained model files containing their weights, stored as `.pth` files.

- `RawData`: Unprocessed synthetic data extracted from multi-agent simulations, stored as `.csv` files.

- `SDD`: Annotations from various different videos from the Stanford Drone Dataset, stored as `.txt` files.

- `Demos`: Video captures of each of the five implemented VMAS scenarios, these show renderings of what the multi-agent simulations used in this project are doing when generating data, stored as `.mp4` files.

Other directories are present in this project which store other facets of the project, these include:

- `ModelTemplates`: Contains all the code defining the structure of the proposed approach, ablation models and the components used to contruct them.

- `ResearchAndDesign`: Contains all information about the project's conception and execution which is not direct implementation. Includes multiple papers, scans and images as well as a journal `Master's Project Journal - JXH1482.pdf` which documents the project's development from 09/10/25 to 27/02/26. The subdirectories are as follows:
  - `Bibliography`: Collection of academic papers/materials used in the research process of this project. None of this is my own work.
  - `Designs`: Scans & images related to the hypothetical and direct implementations of my project's approach.
  - `Research`: Scans of paper notes containing my research insights. Directly sources my bibliography.

- `ExperimentalResults`: Contains all the results produced by the **Temporal Modelling** & **Interaction Modelling** experiments conducted during this project, these are sorted into two subdirectories and contain a `.json` file which holds all the data points produced as well as graph images `.png` depicting these results in a structured format.
