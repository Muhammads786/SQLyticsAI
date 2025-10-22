# Databricks notebook source
# ===========================================================================
# FILE: mlflow_logger.py
# ===========================================================================

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

class MLflowLogger:
    """
    Handle all MLflow logging operations.
    Separates MLflow concerns from model training.
    """
    
    def __init__(self, experiment_name="churn_prediction"):
        mlflow.set_experiment(experiment_name)
        self.run = None
        self.run_id = None
    
    def start_run(self, run_name):
        """Start MLflow run"""
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"✓ MLflow run started: {run_name} (ID: {self.run_id})")
        return self.run
    
    def log_params(self, params_dict):
        """Log parameters"""
        for key, value in params_dict.items():
            mlflow.log_param(key, value)
        print(f"✓ Logged {len(params_dict)} parameters")
    
    def log_metrics(self, metrics_dict):
        """Log metrics"""
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        print(f"✓ Logged {len([v for v in metrics_dict.values() if isinstance(v, (int, float))])} metrics")
    
    def log_artifacts(self, filepaths_dict):
        """Log artifact files"""
        for name, filepath in filepaths_dict.items():
            if filepath:
                mlflow.log_artifact(filepath)
        print(f"✓ Logged {len(filepaths_dict)} artifacts")
    
    def log_model(self, model, train_df, predictions, feature_cols, 
                  registered_model_name="customer_churn_model"):
        """Log Spark ML model"""
        signature = infer_signature(
            train_df.select(feature_cols).toPandas(),
            predictions.select("prediction").toPandas()
        )
        
        mlflow.spark.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=registered_model_name
        )
        print(f"✓ Model logged to registry: {registered_model_name}")
    
    def end_run(self):
        """End MLflow run"""
        if self.run:
            mlflow.end_run()
            print(f"✓ MLflow run ended: {self.run_id}")