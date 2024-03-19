local model_name = "facebook/bart-base";

# These are all overridden if you use train.sh
# paths are from root because we're expecting to be inside a docker container
local train_data_path = "data/processed/shift_reduce/toy/neural_input.txt";   //"toy_dataset/toy-train.tsv";
local validation_data_path = "data/processed/shift_reduce/toy/neural_input.txt";  //"toy_dataset/toy-train.tsv";
# comet (default unused because API is required and we don't want those on GitHub)
local use_comet = false;
local comet_api_key = "";
local comet_project_name = "mg-shift-reduce";

{
  "dataset_reader": {
    "type": "shift_reduce",
  },
  
  "train_data_path": train_data_path,  // defined at the top
  "validation_data_path": validation_data_path,
  
  "model": {
    "type": "neural_shift_reduce", // replace this with a different model in eg https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/ to use other models
    "num_silent_heads": 5,
    "silent_heads_dim": 32,
    "decoder_hidden_dim": 32,
  },
  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": 1,
        "padding_noise": 0.0
    }
  },
  "trainer": {
    "run_confidence_checks": false,  // TODO what is this??
    "num_epochs": 1,
    "patience": null,  // set a number for early stopping
    "validation_metric": "-loss", // -loss is the default
    "cuda_device": -1,  // overridden in train.sh
    "optimizer": {
      "type": "adam",
      "lr": 0.001  // needs to be very small for BART; try 0.00002 (Jonas) (try 0.00001 for real data?)
      },
    "callbacks": [{ "type":"comet",
                    "use_comet": use_comet,
                    "comet_project_name": comet_project_name,
                    "comet_api_key": comet_api_key,
                    "should_log_parameter_statistics": false,  // might reduce the amount of noise in comet metrics
                    "should_log_learning_rate": true
                    }],  // overridden in train.sh
  }
}