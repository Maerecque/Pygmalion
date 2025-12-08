[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/) [![Python 3.7.9](https://img.shields.io/badge/python-3.7.9-blue.svg)](https://www.python.org/downloads/release/python-379/)
<p>
  <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" height="30" alt="CC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" height="30" alt="BY">
  <img src="https://mirrors.creativecommons.org/presskit/icons/nc-eu.svg" height="30" alt="NC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" height="30" alt="SA">
</p>

# Pymalion Pointcloud Converter
This repository will contain the code made for the Utrecht local government. This project will contain the code that will be used for saving the wharf cellar in an organized matter. This Python script is designed for processing point cloud data, primarily in LAS and LAZ file formats. It offers various functionalities for point cloud manipulation, including downsampling, noise removal, plane extraction, repairing, and transformation. Below, you'll find an overview of the script and how to use it effectively.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [Contributing](#contributing)
- [License](#license)

## Requirements [OLD]
Before using this script, ensure you have the following prerequisites installed on your system:

- Python 3.x [tested with versions 3.7.9 and 3.9.7]
- Required Python packages, which can be installed using pip as mentioned [below](#installation):
  - laspy
  - laszip
  - lazrs
  - matplotlib
  - numpy
  - open3d
  - plyfile
  - scikit-learn
  - tk
  - tqdm

## Installation
1. Clone this repository to your local machine or download the script directly.
2. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage in CLI
1. Open a terminal and navigate to the directory where the script is located.
2. Run the script using the following command:

   ```bash
   python main.py
   ```

3. The script will prompt you to select an LAS or LAZ file for processing. Choose the appropriate file.
4. The script will execute a series of point cloud processing operations, and you will be presented with intermediate results and the option to visualize the point cloud data using an external viewer.

## Usage as application
1. Open and execute `prep.cmd`. This will take several minutes, depending on your system specifications.
2. After this step is finished a new file named `Pygmalion.exe` will be created. Open this file to start the application.

## Functionality [OLD]
The script provides the following main functionalities:

1. **Downsampling:** The point cloud is downsampled to reduce the number of points, improving processing speed.

2. **Noise Outlier Removal:** Statistical noise removal is applied to enhance the quality of the point cloud data.

3. **Plane Extraction:** The script divides the point cloud into a 3D grid and identifies planes within the point cloud.

4. **Point Cloud Repair:** Detected planes are used to repair the point cloud data, resulting in a smoother and more coherent point cloud.

5. **Mesh Transformation:** The script can transform the repaired mesh back into a point cloud for further processing or visualization.

6. **PLY Module (Optional):** This section of the script is commented out by default but can be uncommented if you want to save the processed point cloud data as a PLY file.

Please note that the script contains additional commented-out functionality related to PLY file conversion, which can be enabled if needed.


## License
This repository and code is made under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. You are free to share and adapt the material, provided you give appropriate credit, do not use it for commercial purposes, and distribute any modified material under the same license.