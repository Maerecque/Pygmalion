[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/) [![Python 3.7.9](https://img.shields.io/badge/python-3.7.9-blue.svg)](https://www.python.org/downloads/release/python-379/)
<p>
  <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" height="30" alt="CC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" height="30" alt="BY">
  <img src="https://mirrors.creativecommons.org/presskit/icons/nc-eu.svg" height="30" alt="NC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" height="30" alt="SA">
</p>

# Pygmalion Pointcloud Converter
This repository will contain the code made for the Association of Netherlands Municipalities (VNG) by me. This projects main purpose is to process and organize point cloud data for the purpose of saving the wharf cellar in an open and standardized 3d format. This format is to be used for future urban planning and maintenance of the wharf cellar structures.

This application uses LAS/LAZ files as input and process them through various point cloud manipulation techniques such as downsampling, noise removal, plane extraction, repairing and transformation. The processed point cloud data can then be exported in a standardized format for further use.


## Table of Contents
- [Requirements](#requirements)
  - [PC Specifications](#pc-specifications)
  - [Python Environment](#python-environment)
- [Installation](#installation)
  -  [Usage in CLI](#usage-in-cli)
  -  [Usage as executable](#usage-as-executable)
- [Contributing](#contributing)
- [Background](#background)
- [License](#license)


## Requirements
### PC Specifications
This application has only been tested on Windows 10 and Windows 11 systems. Below are the minimum specifications required to run the script effectively, after that the recommended specifications are listed.
#### Minimum Specifications
- CPU: Intel Core i7 8th Gen or AMD equivalent
- RAM: 8GB
- Storage: 256GB SSD
- GPU: Integrated graphics

#### Recommended Specifications
- CPU: Intel Core i7 10th Gen or AMD equivalent
- RAM: 16GB or more
- Storage: 512GB SSD or larger
- GPU: Dedicated graphics card with at least 6GB VRAM

### Python Environment
The script requires the following software and libraries:
- Python 3.x [tested with versions 3.7.9 and 3.9.7]
- The required Python packages can be installed using the provided `requirements.txt` file.


## Installation
1. Clone this repository to your local machine or download the script directly.
2. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Usage in CLI
1. Open a terminal and navigate to the directory where the script is located.
2. Run the script using the following command:

   ```bash
   python main.py
   ```

3. The script will prompt you to select an LAS or LAZ file for processing. Choose the appropriate file.
4. The script will execute a series of point cloud processing operations, and you will be presented with intermediate results and the option to visualize the point cloud data using an external viewer.


### Usage as executable
1. Open and execute `prep.cmd`. This will take several minutes, depending on your system specifications.
2. After this step is finished a new file named `Pygmalion.exe` will be created. Open this file to start the application.


## Contributing
Contributions to this repository are welcome. If you would like to contribute, please fork the repository and create a pull request with your changes. Make sure to follow the existing code style and include appropriate tests for any new functionality. The main branch is protected, so all changes must go through a pull request and be reviewed before being merged. Thank you for your interest in contributing to this project!

## Background
This project was developed for the Association of Netherlands Municipalities (VNG) to address the need for efficient processing and organization of point cloud data related to wharf cellar structures. The goal is to create an open and standardized 3D format that can be used for urban planning and maintenance purposes. The project leverages various point cloud manipulation techniques to ensure high-quality data output suitable for future applications.

Originally this application was made for as my graduation assignment for the HBO-ICT AI at the University of Applied Sciences in Utrecht, The Netherlands. This project has been dormant after my graduation in the summer of 2023, but has been revived in mid 2024 for further development and use by the VNG. Innitially I never intended to make this project open source, but after some consideration I have decided to do so in the hope that others might find it useful or be able to contribute to its further development.

### Name Origin
The name "Pygmalion" is inspired by the Greek myth of Pygmalion, who put his artistic talents to use by sculpting a statue that was so beautiful and lifelike that he fell in love with it. The goddess Aphrodite brought the statue to life in response to his prayers. This name reflects the transformative nature of the application, which takes raw point cloud data and processes it into a refined and usable 3D format, much like how Pygmalion's statue was transformed into a living being.

## License
This repository and code is made under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. You are free to share and adapt the material, provided you give appropriate credit, do not use it for commercial purposes, and distribute any modified material under the same license.