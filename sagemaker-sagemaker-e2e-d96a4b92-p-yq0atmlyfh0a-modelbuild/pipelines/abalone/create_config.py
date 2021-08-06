"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()
    base_dir = "/opt/ml/processing"
    
    analysis_config = {
      "dataset_type": "text/csv",
      "headers": [
        "rings",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "sex_M",
        "sex_I",
        "sex_F"
      ],
      "label": 0,
      "label_values_or_threshold": [
        1
      ],
      "facet": [
        {
          "name_or_index": "sex_M",
          "value_or_threshold": [
            0
          ]
        }
      ],
      "predictor": {
        "model_name": "",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
        "accept_type": "text/csv"
      },
      "probability_threshold": 0.5,
      "methods": {
        "pre_training_bias": {
          "methods": "all"
        },
        "post_training_bias": {
          "methods": "all"
        },
        "report": {
          "name": "report",
          "title": "Analysis Report"
        }
      }
    }
    
    
    analysis_config['predictor']['model_name'] = args.model_name
    
    

    config_file = f"{base_dir}/clarify/analysis_config.json"

    with open(config_file, 'w+') as file:
         file.write(json.dumps(analysis_config))