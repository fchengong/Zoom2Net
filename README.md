# Zoom2Net: Constrained Network Telemetry Imputation
This code corresponds to the paper: **Zoom2Net: Constrained Network Telemetry Imputation**.

## Setup
`cd zoom2net/`

Inside an already *existing* root directory, each experiment will create a time-stamped output directory, which contains
model checkpoints.
The following commands assume that you have created a new root directory inside the project directory like this: 
`mkdir experiments`.

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run Zoom2Net

To see all command options with explanations, run: `python src/main.py --help`

## Train models from scratch

```bash
python src/main.py --output_dir path/to/experiments --window_size 1000 --window_skip 100 --zoom_in_factor 50
```

## Gurobi license

Get lisence from Gurobi User Portal https://portal.gurobi.com/iam/licenses/list.