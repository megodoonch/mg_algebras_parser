local model_name = "facebook/bart-base";

# These are all overridden if you use train.sh
# paths are from root because we're expecting to be inside a docker container
local train_data_path = "/data/seq2seq/official/train/train.tsv";
local validation_data_path = "/data/seq2seq/official/dev/dev.tsv";
//local predictions_path = "/predictions/predicted.txt";  # use when predicting
local predictions_path = null;  # use when training
local use_comet = true;
local comet_api_key = "";  # add your comet API key here
local comet_project_name = "mg-bart";

{
  "dataset_reader": {
    "type": "seq2seq",
    "source_add_start_token": false,  // BART tokenizer seems to add these itself
    "source_add_end_token": false,
    "target_add_start_token": false,
    "target_add_end_token": false,
    "source_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name,
        "namespace": "tokens"
      }
    },
    "source_max_tokens": 1024,  // BART can only do input sequences up to length 1024.
    "target_max_tokens": 1024,
  },
  
  "train_data_path": train_data_path,  // defined at the top
  "validation_data_path": validation_data_path,
  
  "model": {
    "type": "mego_bart", // replace this with a different model in eg https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/ to use other models
    "model_name": model_name,  // defined above
    //"target_namespace": "target_tokens",
    "beam_size": 1,  # make it faster with lower beam size (default 10)
    "max_decoding_steps": 1024,  # default 50 (more is slower): max length of output sentence for BLEU and ROUGE
    "predictions_path": predictions_path,
  },
  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": 4,
        "padding_noise": 0.0
    }
  },
  "trainer": {
    "run_confidence_checks": false,
    "num_epochs": 100,
    "patience": 10,  // set a number for early stopping
    "validation_metric": "-loss", // -loss is the default
    "cuda_device": -1,  // overridden in train.sh
    "optimizer": {
      "type": "adam",
      "lr": 0.00001  // needs to be very small for BART; try 0.00002 (Jonas)
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