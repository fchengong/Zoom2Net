# Zoom2Net: Constrained Network Telemetry Imputation
This code corresponds to the paper: **Zoom2Net: Constrained Network Telemetry Imputation**.
This code has been tested with `Python 3.8` and `Pytorch 2.0`.

## Setup
`cd zoom2net/`

`pip install -r requirements.txt`

## Hardware and Resource Requirements
We train models for all use cases on an Nvidia Tesla T4-16GB GPU. It may take about 20-60min to finish for different cases.

## Run Zoom2Net for different usecases

There are 3 usecases with 4 different datasets in separate directories. Move to each directory to see detailed running instruction.

## Gurobi license

Running Gurobi ILP requires a license. Get license from Gurobi User Portal https://portal.gurobi.com/iam/licenses/list. And place the file `gurobi.lic` to `/opt/gurobi and your home directory (/home/yourusername)` for Linux system. 
