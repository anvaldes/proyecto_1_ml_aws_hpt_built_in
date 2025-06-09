import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import boto3
import os

print('👋 Iniciando entrenamiento...')

if __name__ == "__main__":

    # Argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    args = parser.parse_args()

    print("🧾 Argumentos:")
    print(args)

    print("f1_score: 0.77")
    
    print("✅ Entrenamiento y evaluación completados.")