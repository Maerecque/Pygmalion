[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/) [![Python 3.7.9](https://img.shields.io/badge/python-3.7.9-blue.svg)](https://www.python.org/downloads/release/python-379/) ![version](https://img.shields.io/badge/version-1.0.0-g.svg)
<p>
  <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" height="30" alt="CC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" height="30" alt="BY">
  <img src="https://mirrors.creativecommons.org/presskit/icons/nc-eu.svg" height="30" alt="NC">
  <img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" height="30" alt="SA">
</p>

# Pygmalion Pointcloud Converter
Pygmalion is a point cloud processing application developed for the Association of Netherlands Municipalities (VNG). Its primary goal is to process, clean, and organize point cloud data related to wharf cellar structures, enabling their preservation in an open and standardized 3D format. It is mainly tested on scans of the wharf cellar structures from the city of Utrecht, but should work on other point cloud data as well.

This standardized output is intended to support future urban planning, inspection, and maintenance workflows for municipal infrastructure.

This application uses LAS/LAZ files as input and process them through various point cloud manipulation techniques such as downsampling, noise removal, plane extraction, repairing and transformation. The processed point cloud data can then be exported in a standardized format for further use.

## Key Features
- Input support for LAS/LAZ point cloud formats
- Point cloud processing operations, including:
  - Downsampling
  - Noise removal
  - Plane extraction
  - Data repair
  - Coordinate transformation
- Export to a standardized 3D format suitable for long-term use
- Optional visualization of intermediate and final results

---

## Table of Contents
- [Requirements](#requirements)
  - [PC Specifications](#pc-specifications)
  - [Python Environment](#python-environment)
- [Installation](#installation)
  -  [Usage in CLI](#usage-in-cli)
  -  [Usage as executable](#usage-as-executable)
- [Contributing](#contributing)
- [Background](#background)
  - [Name Origin](#name-origin)
- [License](#license)


## Requirements
### PC Specifications
This application has been tested on Windows 10 and Windows 11 only.

#### Minimum Specifications
- CPU: Intel Core i7 (8th Gen) or AMD equivalent  
- RAM: 8 GB  
- GPU: Integrated graphics

#### Recommended Specifications
- CPU: Intel Core i7 (10th Gen) or AMD equivalent  
- RAM: 16 GB or more  
- GPU: Dedicated GPU with at least 6 GB VRAM

### Python Environment
The script requires the following software and libraries:
- Python 3.x 
    - Tested with versions 3.7.9 and 3.9.7
- Required Python packages are listed in `requirements.txt`

---
## Installation
1. Clone this repository or download the source code.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---
### Usage in CLI
1. Open a terminal and navigate to the project directory.
2. Run the application:
   ```bash
   python main.py
   ```
3. Select an LAS or LAZ file when prompted.
4. The application will execute a sequence of point cloud processing steps.
5. Intermediate results can optionally be visualized using an external viewer.
---
### Usage as executable
1. Run `prep.cmd`
   - This step will take several minutes.
   - There will be some console output during the process; please wait until it finishes.
   - Your computer may prompt you to allow the program to make changes; please allow it.
2. After completion, a file named `Pygmalion.exe` will be generated.
   - Please note that your settings for presets will be fixed in the generated executable. To change them, you will need to re-run `prep.cmd`.
3. Launch `Pygmalion.exe` to start the application.
---

## Contributing
Contributions to this repository are welcome. If you would like to contribute, please fork the repository and create a pull request with your changes. Make sure to follow the existing code style and include appropriate tests for any new functionality. The main branch is protected, so all changes must go through a pull request and be reviewed before being merged. Thank you for your interest in contributing to this project!

## Background
This project was originally developed as my graduation assignment for my studies in HBO-ICT Artificial Intelligence at the University of Applied Sciences in Utrecht, The Netherlands. After my graduation in the summer of 2023, the development of this project was dormant for a period but was revived in mid-2024 for further development with the goals of the Association of Netherlands Municipalities (VNG) in mind.

Innitially I never intended to make this project open source, but as per discussions with the VNG and considering the potential benefits for other municipalities and organizations, I decided to release it under an open-source license to encourage collaboration and further development.

### Name Origin
The name Pygmalion is inspired by the Greek myth of Pygmalion, who put his artistic talents to use by sculpting a statue that was so beautiful and lifelike that he fell in love with it. The goddess Aphrodite brought the statue to life in response to his prayers. This name reflects the transformative nature of the application, which takes raw point cloud data and processes it into a refined and usable 3D format, much like how Pygmalion's statue was transformed into a living being.

## License
This repository and code is made under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. You are free to share and adapt the material, provided you give appropriate credit, do not use it for commercial purposes, and distribute any modified material under the same license.
