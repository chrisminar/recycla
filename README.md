# Recycla

## setup
> cd <source directory>
> python -m venv .venv
> source .venv/Scripts/activate
> pip install -r requirements.txt
> pip install -e recycla

The requirements.txt file imports the non-cuda version which trains slowly but is much smaller.
should install cuda for faster training
you will need to install cuda, then get the appropriate torch version
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Sanity check
> pytest

## Basic training

1. Download data from X TODO
2. Rename data X TODO
3. Generate labeled data
> recycla parse-data
4. Train (recomended to setup CUDA before training)
> recycla train --model-type mobilenet_v2 --freeze-backbone --unfreeze-last-n-blocks 2 --nepochs 3  

Freezing all but the last 2 blocks will dramatically improve training speed. I typically saw this loose accuracy.  
The output will be under .models/best_candidate.pth
> mv .models/best_candidate.pth .models/best_mobilenetv2.pth
5. Evaluate
> recycla test image --model-path .models/best_mobilenetv2.pth --save-path .results/mobilenet
6. Train a different model
7. Compare
> recycla test image --model-path .models/best_mobilenetv2_experiment2.pth --save-path .results/mobilenet_experiment2
> recycla compare .results/mobilenet .results/mobilenet_experiment2

## Available commands

Make sure recycla is installed in your venv (see setup)

#### See all commands
> recycla --help

#### Split raw data into training, test, and validation datasets
> recycla parse-data

#### Count the training files
> recycla count

#### Train a new model
> recycla train --model-type mobilenet_v2  

There are several common models implmented for training. I recommend you start with mobilnetv2, which is the default. It is small, fast, and gets good results.
Then try efficientnet, I got very good results with it.
> recycla train --model-type efficientnet_v2_s  

Note: to not overwrite the first model you will need rename the generate .pth file to something else

#### Evaluate model
> recycla test image

#### Compare two outputs
> recycla compare <results_path_1> <results_path_2>
Note to non-default the results comparison path you will have to run it like this
> recycla test image --model-path <model_path.pth> --save-path <results_path>


## Trouble shooting
Fix `command not found: recycla` with  
> cd <recycla_dir>
> source .venv/bin/activate
> `pip install -e recycla`
