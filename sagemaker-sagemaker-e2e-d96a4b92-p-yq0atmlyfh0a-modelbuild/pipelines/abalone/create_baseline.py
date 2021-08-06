import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker>=2.48"])

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import argparse 
import sys

if __name__ == '__main__':
    ''' '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-data-uri", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)  
    parser.add_argument("--role", type=str, required=True)  
    args, _ = parser.parse_known_args()
    
    baseline_prefix = args.prefix + "/data_quality_baselining"
    baseline_results_prefix = baseline_prefix + "/results"
    baseline_results_uri = "s3://{}/{}".format(args.bucket, baseline_results_prefix)
    role = args.role
    
    data_quality_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )

    data_quality_monitor.suggest_baseline(
        baseline_dataset=args.baseline_data_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        wait=True,
    )
