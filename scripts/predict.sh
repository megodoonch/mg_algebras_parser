# use this inside a docker container
# for baseline models

input_path=$1
bart_or_lstm=$2

if [ "$bart_or_lstm" == "bart" ]; then
  module=bart_for_predict
else module=my_seq2seq
fi

subdir=$(date +"%Y-%m-%d-%H-%M")  #$(date +%s)
log_dir=/logfiles/$bart_or_lstm/$subdir

cd ../mounted || exit

echo "making log directory at $log_dir"
mkdir -p "$log_dir"

echo "running allennlp predict..."

allennlp predict /models/model.tar.gz "$input_path" --use-dataset-reader --include-package minimalist_parser.neural.$module > "$log_dir"/log

echo "changing file permissions..."

chmod 777 -R /logfiles
chmod 777 -R /predictions

#echo "copying predictions to storage volume"
#cp results/predictions/bart/predictions.txt /predictions/bart/predictions.txt
