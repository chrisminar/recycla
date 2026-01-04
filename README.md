# Recycla

## Setup
```bash
cd <source directory>
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e recycla
```

The requirements.txt file imports the non-CUDA version which trains slowly but is much smaller.
You should install CUDA for faster training.
You will need to install CUDA, then get the appropriate PyTorch version:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Sanity check:
```bash
pytest
```

## Basic Training

1. Download data from X TODO
2. Rename data X TODO
3. Generate labeled data
   ```bash
   recycla parse-data
   ```
4. Train (recommended to set up CUDA before training)
   ```bash
   recycla train --model-type mobilenet_v2 --freeze-backbone --unfreeze-last-n-blocks 2 --nepochs 3
   ```

   Freezing all but the last 2 blocks will dramatically improve training speed. I typically saw this lose accuracy.
   The output will be under `.models/best_candidate.pth`
   ```bash
   mv .models/best_candidate.pth .models/best_mobilenetv2.pth
   ```
5. Evaluate
   ```bash
   recycla test image --model-path .models/best_mobilenetv2.pth --save-path .results/mobilenet
   ```
6. Train a different model
7. Compare
   ```bash
   recycla test image --model-path .models/best_mobilenetv2_experiment2.pth --save-path .results/mobilenet_experiment2
   recycla compare .results/mobilenet .results/mobilenet_experiment2
   ```

## Available Commands

Make sure recycla is installed in your virtual environment (see setup)

#### See all commands
```bash
recycla --help
```

#### Split raw data into training, test, and validation datasets
```bash
recycla parse-data
```

#### Count the training files
```bash
recycla count
```

#### Train a new model
```bash
recycla train --model-type mobilenet_v2
```

There are several common models implemented for training. I recommend you start with MobileNetV2, which is the default. It is small, fast, and gets good results.
Then try EfficientNet, I got very good results with it.
```bash
recycla train --model-type efficientnet_v2_s
```

Note: To not overwrite the first model, you will need to rename the generated .pth file to something else.

#### Evaluate model
```bash
recycla test image
```

#### Compare two outputs
```bash
recycla compare <results_path_1> <results_path_2>
```
Note: To use a non-default results comparison path, you will have to run it like this:
```bash
recycla test image --model-path <model_path.pth> --save-path <results_path>
```


## Troubleshooting
Fix `command not found: recycla` with:
```bash
cd <recycla_dir>
source .venv/bin/activate
pip install -e recycla
```
