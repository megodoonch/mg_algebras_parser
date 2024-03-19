#!/bin/bash

Help()
{
   # Display Help
   echo "Trains a model. Defaults depend on whether you're training a toy or real model. (Default real)"
   echo
   echo "Syntax: bash train.sh [-s] [-g] [-e num_epochs] [-a comet_api_key] [-p comet_project_name] [-t training_corpus_path] [-d dev_corpus_path] [-m model_path] [-c confile_file_path]"
   echo "options:"
   echo "-n    don't log to comet.ml (flag)        (default: use comet)"
   echo "-s    use toy data set (flag)             (default: use real data set)"
   echo "-a    comet API key for logging to comet  (default: given in config file)"
   echo "-p    comet project to log to             (default: mg-seq2seq or mg-bart depending on whether -b flag is used)"
   echo "-t    path to training corpus .tsv file   (default official split for real, toy for toy)"
   echo "-d    path to validation corpus .tsv file"
   echo "-m    path to store trained model         (default trained_models/models/bart|seq2seq/official|toy/timestamp)"
   echo "-f    path to copy trained model          (default don't copy)"
   echo "-e    number of epochs to train for       (default 1000 for official; 2 for toy)"
   echo "-c    path to trainer config file         (default config_files/official_seq2seq|bart.jsonnet)"
   echo "-g    use GPU (flag)                      (default false)"
   echo "-b    bart model (flag)                   (default false, so seq2seq)"
   echo "-h    Help (flag)"
}


# Defaults
use_comet=true
gpu=false
comet_callback=minimalist_parser.neural.comet_callback
prefix="official"
model="seq2seq"

while getopts "e:t:d:a:p:e:m:f:c:s:gbhn" opt; do
    case $opt in
	 h) Help
	   exit
	   ;;
	 n) use_comet=false
	   ;;
	 c) config_file_path="$OPTARG"
	   ;;
	 t) train_data_path="$OPTARG"
	   ;;
	 d) validation_data_path="$OPTARG"
	   ;;
	 m) model_path="$OPTARG"
	   ;;
	 f) final_model_path="$OPTARG"
	   ;;
	 a) comet_api_key="$OPTARG"
	   ;;
	 p) comet_project_name="$OPTARG"
	   ;;
	 e) num_epochs="$OPTARG"
	   ;;
	 g) gpu=true
	   ;;
	 s) prefix="$OPTARG"
	   ;;
	 b) model="bart"
	   ;;
	 \?) echo "Invalid option -$OPTARG" >&2
	    ;;
	 esac
done


comet_project_name="mg-$model-silent-heads"
model_python_file=minimalist_parser.neural.my_$model


# if no config file is given, use toy_bart.jsonnet or original_bart.jsonnet
if [ -z "${config_file_path+x}" ]; then
  config_file_path="config_files/${prefix}_$model.jsonnet"
fi
echo "config_file_path: ${config_file_path}"


if  [ -z ${model_path+x} ]; then
  subdir=$(date +"%Y-%m-%d-%H-%M")  #$(date +%s)
  model_path="trained_models/models/$model/${prefix}/$subdir/"
fi

echo "model path: $model_path"
# remove the existing models if any
rm -rf "$model_path"



########### build the overrides ################

overrides="{"
if $use_comet && [ "${comet_project_name}" ] && [ "${comet_api_key}" ]; then
  # build whole callback because it's in a list
  callbacks="[{'type':'comet', 'use_comet': true, 'should_log_parameter_statistics': false, 'should_log_learning_rate': true, "
  if [ "$comet_project_name" ]; then
    callbacks="${callbacks}'comet_project_name': '$comet_project_name', "
    else callbacks="${callbacks}'comet_project_name': comet_project_name, "
  fi
  if [ "$comet_api_key" ]; then
    callbacks="${callbacks}'comet_api_key': '$comet_api_key'}]"
    else callbacks="${callbacks}'comet_api_key': 'comet_api_key'}]"
  fi
    overrides="${overrides}'trainer.callbacks': $callbacks,"
elif ! $use_comet; then
    callbacks="[{'type':'comet', 'use_comet': false,}]" # 'comet_project_name': ${comet_project_name}', 'comet_api_key: XXX', 'should_log_parameter_statistics': false, 'should_log_learning_rate': true}]"
    overrides="${overrides}'trainer.callbacks': ${callbacks},"
fi

if [ "${train_data_path}" ]; then
  overrides="$overrides'train_data_path': '$train_data_path', "
fi
if [ "${validation_data_path}" ]; then
  overrides="$overrides'validation_data_path': '$validation_data_path', "
fi
if $gpu; then
  echo "requesting GPU"
  overrides="$overrides'trainer.cuda_device': 0, "
fi
if [ "${num_epochs}" ]; then
  overrides="$overrides'trainer.num_epochs': $num_epochs, "
fi

if [ "$overrides" != "{" ]; then
  overrides=${overrides}"}"
fi


############# TRAIN #############

# CUDA_LAUNCH_BLOCKING=1  prefix this to help with debugging

export TOKENIZERS_PARALLELISM=false
cmd="allennlp train $config_file_path -s $model_path --include-package $model_python_file --include-package $comet_callback"

if [ "$overrides" != "{" ]; then
  cmd="$cmd --overrides $overrides"
  echo "evaluating command $cmd"
  allennlp train $config_file_path -s $model_path --include-package $model_python_file --include-package $comet_callback --overrides "$overrides"
  else
    echo "evaluating command $cmd"
    allennlp train $config_file_path -s $model_path --include-package $model_python_file --include-package $comet_callback
fi

#eval "$cmd" # DO NOT DO THIS!!

ls "$model_path"

echo "changing file permissions..."

chmod 777 -R "$model_path"
chmod 777 -R runs
chmod 777 -R log
#chmod 777 -R /predictions

if [ "${final_model_path}" ]; then
  echo "moving model to $final_model_path"
  mkdir -p "$final_model_path"
  mv "$model_path" "$final_model_path"
#  mv results/predictions/current_predictions.txt "$final_model_path"
  chmod 777 -R "$final_model_path/"
fi