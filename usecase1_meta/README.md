# Usecase 1: ToR burstiness in a Cloud using Meta's dataset

This code has been tested with `Python 3.8` and `Pytorch 2.0`.

To see all command options with explanations, run: `python3 main.py --help`

## Download data
To download the data, run

```bash
git clone https://github.com/facebookresearch/Millisampler-data.git datasets/Millisampler/
```

## Train models from scratch

```bash
python main.py --save_model_dir ./checkpoints/model.torch --task train --window_size 1000 --zoom_in_factor 50
```
note: UserWarning is fine. It will be removed later for optimization purposes. 

In the above command, `model_dir` is the path to save the trained mode; `task` is to denote training a model from scratch; `window_size` is the length of time series to impute each time; `zoom_in_factor` is the ratio between fine-grained time series and coarse-grained time series. Please refer to `python3 main.py --help` for more parameters that can be set. 

We train the model on an Nvidia Tesla T4-16GB GPU. It take about 20min to finish. If you do not want to wait, we also provide a trained model in the next section that reproduces the results in the paper.

## Perform burstiness analysis

```bash
python main.py --z2n_model_dir ./checkpoints/z2n_model.torch --task eval_downstream_task --compute_baselines False
```

In the above command, `z2n_model_dir` is the path to the trained Zoom2Net model; `task` is to denote evaluating downstream tasks (i.e. bursts analysis); getting results from baselines mentioned in the paper (i.e. KNN, training Brits from scratch) can take up to an hour, so we preload the evaluation data using `--compute_baselines False`. If you want to train the baselines from scratch, use `--compute_baselines True`. 

The command will produce results shown in Figure 8b and Figure 9b in the paper. 

## Perform running time analysis
```bash
python main.py --z2n_model_dir ./checkpoints/z2n_model.torch --task eval_timing
```

This command will load the pretrained model with window_size of 1000, zoom-in factor of 50 and output the average inference time. This will produce results in Table 1 in the paper.

## Perform zoom-in factor analysis

```bash
python main.py --task eval_zoom_in_factor
```

This command will load the pretrained models for zoom-in factors of 25, 50, 100 and run downstream tasks for each of them. This will produce results in Figure 11 in the paper. 