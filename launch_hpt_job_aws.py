import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import (
    IntegerParameter,
    HyperparameterTuner
)

# Config

job_name = "xgboost-hpt-sklearn-3"
bucket = "proyecto-1-ml"
source_dir = "."  # directorio donde está train.py y requirements.txt

region = "us-east-1"
role = "arn:aws:iam::613602870396:role/SageMakerExecutionRole"
entry_point = "train.py"
output_path = f"s3://{bucket}/output"

# Hiperparámetros constantes

static_hyperparams = {
    "year": "2025",
    "month": "6"
}

# Rango de hiperparámetros

hyperparameter_ranges = {
    "n_estimators": IntegerParameter(2, 10),
    "max_depth": IntegerParameter(2, 10)
}

# Estimador usando imagen de sklearn

sklearn_estimator = SKLearn(
    entry_point = entry_point,
    source_dir = source_dir,
    role = role,
    instance_type = "ml.m5.large",
    instance_count = 1,
    framework_version = "1.2-1",  # versión oficial de sklearn en SageMaker
    py_version = "py3",
    hyperparameters = static_hyperparams,
    output_path = output_path,
    sagemaker_session = sagemaker.Session(),
    dependencies = ["requirements.txt"]
)

# Tuner
tuner = HyperparameterTuner(
    estimator = sklearn_estimator,
    objective_metric_name = "f1_score",
    objective_type = "Maximize",
    hyperparameter_ranges = hyperparameter_ranges,
    metric_definitions = [
        {"Name": "f1_score", "Regex": "f1_score: ([0-9\\.]+)"}
    ],
    max_jobs = 4,
    max_parallel_jobs = 2,
    base_tuning_job_name = job_name
)

# Lanzar job
tuner.fit(job_name=job_name)

print("✅ HPT job lanzado correctamente sin imagen personalizada.")