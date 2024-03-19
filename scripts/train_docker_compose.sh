# Usage: bash train_docker_compose.sh toy|official home|surfsara

Help()
{
   # Display Help
   echo "Trains a BART model."
   echo
   echo "Syntax: bash train_docker_compose.sh toy|official home|surfsara"
   echo "arguments:"
   echo "1     toy or official"
   echo "      toy for toy data set"
   echo "      official for official data split"
   echo "2     home or surfsara"
   echo "      home for home computer"
   echo "      surfsara for surfsara server"
   echo "3     seq2seq or bart"
#   echo "3     path to logfile directory"
   echo "-h    Help (flag)"
}

while getopts "h" opt; do
    case $opt in
	 h) Help
	   exit
	   ;;
	 *) echo "invalid option"
	 	 esac
done

export LABEL=$1  # toy or official; used by docker compose
export TOKENIZERS_PARALLELISM=false  # doesn't seem to work

location=$2
model=$3

echo "Using docker-compose-train-$location-$model.yml and $1_$model.jsonnet"

if [ $location = "surfsara" ]; then
  logdir="/home/mfowlie/data/volume_2/logfiles/train/$model"
  else
    logdir=logfiles/train/$model
fi

sudo mkdir -p "$logdir"
sudo chmod 777 -R "$logdir"


docker compose -f config_files/docker-compose-train-"$location"-$model.yml up 2>&1 | tee "$logdir/training.log"

