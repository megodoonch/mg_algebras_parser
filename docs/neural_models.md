# About all the neural stuff

**WARNING**: 

* recently reorganised files. There are probably some problems with imports still. 
* Python scripts need to be run with `PYTHONPATH=./` preposed to them. Not tested yet with allennlp

## Set up

###  Quick start with pip:

I don't know if this makes sense anymore, but I'm making a conda environment and then installing everything with pip

```bash
conda create -n env_name python=3.9.12
conda activate env_name
pip install -r requirements.txt
```

### Starting over with pip

This is specifically with cuda toolkit version 11.3, as the new GPUs on Surfsara have `sm_86` something (architechture?) and 10 isn't compatable.
Conda doesn't seem to allow me to install cuda toolkit on my local machine, but pip does.

Check cuda version: this should get 11.3. If you get None, I think that means your torch doesn't have cuda toolkit even if you asked for it.

```python
import torch
print(torch.version.cuda)
```

Make a new conda environment for some reason, and specify the python version.
pip install pytorch instructions here: https://pytorch.org/get-started/locally/

```bash
conda create -n new_env_name python=3.10
conda activate new_env_name
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install allennlp
pip install allennlp-models
```

If you want a docker image with exactly these packages: 

1. pipe pip freeze to requirements.txt 
2. remove any weird @file.... things
3. remove
    ```
    torch==1.12.1+cu113
    torchaudio==0.12.1+cu113
    torchvision==0.13.1+cu113
    ```
4. and make a new docker image



### More Docker

To tag the image with multiple tags or give them multiple names use multiple `-t <name>`s

and to give the container your own name, e.g. `mg-parser-container`: 

```bash
docker build --progress=plain -t mg-parser:latest -t mg-parser:0.x . --no-cache
docker run -it --name mg-parser-container mg-parser
```

To retag or rename:

```bash
docker tag oldname:oldtag newname:newtag
```

In particular, to rename so that it can be pushed to dockerhub:

```bash
docker tag mg-parser:my_tag megodoonch/mg-parser:my_tag
docker push megodoonch/mg-parser:my_tag
```

### Other

To get the requirements from a working environment:

```bash
pip list --format=freeze > requirements.txt
```

To pipe all output to a logfile and also keep it in the terminal, put this at the end of the command:

```bash
 2>&1 | tee path_to_logfile
```



Current Surfsara url: ssh://mfowlie@145.38.188.128

## Setting up Surfsara

* Docker: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04
* Docker compose: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04
* git clone repo
  ```bash
  git config --global credential.helper store
  git clone https://github.com/megodoonch/shift-reduce.git
  ```
* Get mg-parser docker image
  ```bash
  docker login --username=megodoonch
  docker pull megodoonch/mg-parser
  ```

* Enable GPU sometimes?? Today this was necessary and this worked: https://askubuntu.com/questions/1400476/docker-error-response-from-daemon-could-not-select-device-driver-with-capab



## Getting started: parsers

Models require allennlp. Currently this works with python 3.9 and allennlp 2.9. Changing some packages' version leads to disaster, so to be safe, use the packages and versions listed in `requirements.txt`

### Training quick start with Docker image pulled from Dockerhub

Pull the latest from dockerhub and run the image interactively. Pull latest repo from Git.

```bash
docker login --username=megodoonch
docker pull megodoonch/mg-parser
```


**Valid as of July 2023**:

For BART baseline, use Docker Compose, called via a bash script. Pick toy or offical and home or surfsara.

```bash
bash train_docker_compose.sh toy|official home|surfsara seq2seq|bart
```

On the toy data set at home:


```bash
bash train_docker_compose.sh toy home
```


On the official split of the real data set on surfsara:

```bash
bash train_docker_compose.sh official surfsara
```



### Using Docker, not Docker Compose

If you want to run it interactively: `docker run -it megodoonch/mg-parser`

Enable GPU: https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html

With mounted volumes on Surfsara and GPU enabled:

```bash
docker run --gpus all -v ~/raw_data/volume_2/corpora:/app/corpora -v ~/raw_data/volume_2/logfiles:/app/logfiles -v ~/raw_data/volume_2/models:/app/models -it megodoonch/mg-parser
```

```bash
docker run -v ~/PycharmProjects/minimalist_parser/corpora:/app/corpora -v ~/PycharmProjects/minimalist_parser/logfiles:/app/logfiles -v ~/PycharmProjects/minimalist_parser/models:/app/models -it megodoonch/mg-parser
```

### Quick start with Docker using BuildKit:

```bash
export DOCKER_BUILDKIT=1
docker build --progress=plain -t mg-parser:latest .
docker run -it mg-parser
```

The above activates buildkit, builds an image, and runs it in interactive mode.

You can also permanently turn on buildkit by modifying the file `~/.docker/daemon.json` to include the entry `"features": { "buildkit": true }` and restarting the daemon.

If you get an error like `ERROR: rpc error: code = Unknown desc = error getting credentials - err: exec: "docker-credential-desktop": executable file not found in $PATH` then check the file ~/.docker/config.json. If it has an entry `"credsStore" : "desktop"`, change it to `"credStore" : "desktop"`

If you need to run it without buildkit, remove all the `--mount=type=cache,target=...` bits before the actual commands in Dockerfile.

#### Push new image to Dockerhub

log in, add tag username/reponame

```bash
docker login
docker tag mg-parser megodoonch/mg-parser
docker push megodoonch/mg-parser
```


## Models


### Training

See quick start above, in which docker compose uses `docker-compose-train-home|surfsara.yml`, which calls `train.sh`, which:

1. updates the configuration in th YAML file if necessary
2. runs allennlp train
3. changes file permissions and locations so I can access them too

#### What is configured where?

* `official_bart.jsonnet` and `toy_bart.jsonnet`: allennlp config files that control the parameters and the data paths
* `docker-compose-train-home|surfsara.yml`" docker compose config files that control the real locations of the volumes
* `toy` or `official` is passed as an environment variable to docker compose, which determines the choice of YAML file
* `home` or `surfsara` determines the choice of YAML file for docker compose
* comet is hard-coded into the YAML files in local variables at the top
* `my_bart.py` vs `bart_for_predict.py`: Use the latter when predicting because it'll print predictions to file.




#### Training without docker compose

Update the paths in `train.sh` and try:

Train a model with all default values: BART on the toy corpus without comet

```bash
bash train.sh
```

See what your options are:

```bash
bash train.sh -h
```

```bash
bash mkdir -p logfiles/toy/bart/ &&  train.sh  2>&1 | tee logfiles/toy/bart/training.log

```


### Running BART on toy corpus


### Running seq2seq on toy corpus
Update paths in `config_files/mego_seq2seq2.jsonnet` file if necessary:

```
 "train_data_path": "toy_dataset/toy-train.tsv",
 "validation_data_path": "toy_dataset/toy-train.tsv",
```

Make `models` folder if necessary. Run:

```bash
rm -rf models/toy_seq2seq && allennlp train config_files/mego_seq2seq2.jsonnet -s models/toy_seq2seq/ --include-package allennlp_models --include-package my_seq2seq --include-package comet_callback
```

And then it just trains!

On the cloud:

To remove old model:

```bash
rm -rf ../../data/volume_2/models/seq2seq
```

or to rename it e.g. to have a suffix:

```bash
mv ../../data/volume_2/models/seq2seq ../../raw_data/volume_2/models/seq2seq_old
```

To activate conda environment:

```bash
conda activate parser
```

To train:

```bash
allennlp train config_files/seq2seq_cloud.jsonnet -s ../../raw_data/volume_2/models/seq2seq/ --include-package allennlp_models --include-package my_seq2seq --include-package comet_callback
```

Remove old and train

```bash
rm -rf ../../raw_data/volume_2/models/seq2seq && allennlp train config_files/seq2seq_cloud.jsonnet -s ../../raw_data/volume_2/models/seq2seq/ --include-package allennlp_models --include-package my_seq2seq --include-package comet_callback
```

Toy:

```bash
rm -rf ../../raw_data/volume_2/models/toy && allennlp train config_files/seq2seq_cloud.jsonnet -s ../../raw_data/volume_2/models/toy/ --include-package allennlp_models --include-package my_seq2seq --include-package comet_callback
```

#### Troubleshooting and Errors

`OSError: [Errno 27] File too large` means you don't have enough memory allotted

Freezing probably means you used up all the computer's memory. Restart the computer.
To prevent this issue in future, give yourself a limited amount of memory. I've been finding with 8GB of RAM I can give myself 6GB, but to train the BART model I need to close PyCharm. 

```bash
ulimit 6000
```

### allennlp files copied and updated

* copied `simple_seq2seq` over to `my_seq2seq`
  * new model is called `MegoSeq2Seq`
  * Added `self._accuracy` and it added to `get_metrics()` so it's loggable.
  * Added a function `_compute_accuracy` modelled on `_get_loss` that ignores the start symbol and then calculates accuracy
  * accuracy is calculated when loss is calculated, in `_forward_pass`

* copied `bart.py` over to `my_bart.py` and `bart_for_predict.py`
  * new model is called `MegoBart` registered as `mego_bart`
  * calculates accuracy instead of blue and rouge
  * `bart_for_predict` prints predictions to file
  * `my_bart` prints predictions to file while training if you give it a `predictions_path` parameter, defined in the YAML file



#### Comet ML logging

`allennlp` already has a Callback module called `allennlp.training.callbacks.tensorboard`.
Meanwhile, Comet has added integration with TensorboardX (https://colab.research.google.com/drive/1cTO3tgZ03nuJQ8kOjZhEiwbB-45tV4lm?usp=sharing#scrollTo=Ocstx_ZafPZm)
(However, I found I had to do some things differently from in the Notebook.)

To log to Comet during training, I wrote a subclass of `TensorBoardCallback` called `CometCallback`, in `comet_callback.py`. 
It overrides very little. The `init` function has 4 new optional parameters:

```python
use_comet: bool = False,
comet_api_key: str = None,
comet_workspace_name: str = None,
comet_project_name: str = None,
```

It makes a `comet_config` dictionary, and then initialises the `SummaryWriter` with `comet_config` instead of directories.
(This parameter is also gone, and the empty string passed to `super()`.)
It logs everything to the same writer, otherwise we get two comet projects.

`GradientDecentTrainer` has an optional `callbacks` argument, so I add to the config file, under `trainer`:

```jsonnet
"callbacks": [{ "type": "comet", 
                "use_comet": true,
                "comet_project_name": comet_project_name,
                "comet_api_key": comet_api_key,
                "should_log_parameter_statistics": true,
                "should_log_learning_rate": true}]
```

where `comet_project_name` and `comet_api_key` are defined as local variables, e.g.

```jsonnet
local comet_project_name = "Toy";
```

Finally,
`--include-package comet_callback` is added to the command that calls the trainer.

Note that contra the comet ML tutorial, I never call `comet_ml.init()`. Initialising the Writer seems to do that on its own.

###### Version Dependencies

`tensorboardX` is an opensource version of `tensorboard`, which logs kind of like comet, but maybe just for Tensorflow. 
In any case, `tensorboardX` works with `PyTorch`, so we can use it. Need tensorboardx version higher than 2.2. 2.4 works and is available in `pip`, but maybe not conda.

`allennlp` needs version 2.9.1, which fixes a bug, but that bug was introduced late, so maybe an earlier version works too. (Just not 2.9.0)


#### Predicting

call `allennlp predict` with path to model then path to dataset to predict from. `--use-dataset-reader` means to use the
same dataset reader as in the training, which we need because our data is in TSV, not JSON, which is what the default reader reads.

I made a script and a docker-compose config file for prediction. Docker for some reason suddenly has serious permissions problems, so the script gives 777 permissions on the created files.

```bash
docker compose -f config_files/docker-compose-predict-home.yml up
```

Note that docker compose has a bug that makes it impossible to tab-complete on the path to the config file.

To change the path to the input file, change it in config_files/docker-compose-predict-home.yml

At home on toy dataset:

```bash
allennlp predict models/toy_seq2seq/model.tar.gz toy_dataset/toy-train.tsv --use-dataset-reader --include-package my_seq2seq
```

Above code generates files with sentences in the same order as the gold files, so we can easily view them side by side.
Print prediction and gold for sentence n:

```bash
python allennlp_input_codec.py path/to/predictions path/to/gold n
```


### BART

Change the `"model"` values in the config file to be `"bart"` and update all the sub-values to match the parameters of the Bart class. You need to follow the trail of parameter types, so if bart needs a parameter of type X, X might also need its parameter values filled in the config file. Parameters with default values don't usually need to be filled.

See [allennlp models page](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/) for `bart.py`

TODO: Running the line below works, but I've suppressed a warning/error with `"run_confidence_checks": false`

```
rm -rf models/toy_bart && allennlp train config_files/bart.jsonnet -s models/toy_bart/ --include-package allennlp_models
```

With new config file:

```bash
rm -rf models/toy_bart && allennlp train config_files/mego_bart.jsonnet -s models/toy_bart/ --include-package my_bart --include-package comet_callback
```

### Evaluation

#### Printing predictions

There are predict scripts in scripts, one for surfsara and one for home. They're designed to be run inside a docker container. To do this, use docker compose:

On the server:

```bash
docker compose -f config_files/docker-compose-predict-surfsara.yml up
```

At home:

```bash
docker compose -f config_files/docker-compose-predict-home.yml up
```

With the following command (for bart or seq2seq), we can print the predictions of the trained model on the original training set. 
It prints to stdout and is hard to read, but we can probably change it to print to file.

We give it the path to the trained model and the path to the dataset to predict. The dataset needs to have the same format as the training set because we're using the `--use-dataset-reader` flag.

```
allennlp predict models/toy_bart/model.tar.gz toy_dataset/toy-train.tsv --include-package allennlp_models --use-dataset-reader
allennlp predict models/toy_seq2seq/model.tar.gz toy_dataset/toy-train.tsv --include-package allennlp_models --use-dataset-reader
```

Notes:

* seq2seq gives top 10 results, and bart gives top 1
* Useful? https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
  * might need to truncate trees first

