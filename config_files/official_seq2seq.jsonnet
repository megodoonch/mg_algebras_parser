local train_data_path = "/data/seq2seq/official/train/train.tsv";   //"toy_dataset/toy-train.tsv";
local validation_data_path = "/data/seq2seq/official/dev/dev.tsv";  //"toy_dataset/toy-train.tsv";
local use_comet = true;
local comet_api_key = "";  # add your comet API key here
local comet_project_name = "mg-seq2seq";
local dropout = 0.5;  // 0 for none


{
  "dataset_reader": {
    "type": "seq2seq",
    "source_tokenizer": {
      "type": "whitespace"
    },
    "target_tokenizer": {
      "type": "whitespace"
    },
	// not sure how the indexers work / what they do / if this is right
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    }
  },
  
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  
  "model": {
    "type": "mego_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 100,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 512, // try 128 if nec
      "num_layers": 2, // need at least 2 if using dropout
      "dropout": dropout,

    },
    "target_namespace": "target_tokens",
    "beam_size": 1,  # make it faster with lower beam size
    "max_decoding_steps": 1040,  # (more is slower): max length of output sentence
  },
  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": 20,
        "padding_noise": 0.0
    }
},
  "trainer": {
    "num_epochs": 50,
    "patience": null,  // stop after n epochs without improvement on dev
    "validation_metric": "-loss", // -loss is the default
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },

    "callbacks":[{ "type":"comet",
                    "use_comet": use_comet,
                    "comet_project_name": comet_project_name,
                    "comet_api_key": comet_api_key,
                    "should_log_parameter_statistics": true,
                    "should_log_learning_rate": true}],
    }
}