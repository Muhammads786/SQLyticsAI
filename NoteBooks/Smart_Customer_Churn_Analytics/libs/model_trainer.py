# Databricks notebook source
# ===========================================================================
# FILE: model_trainer.py
# ===========================================================================

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import numpy as np

class MLModelTrainer:
    """
    Train different ML models with consistent interface.
    Supports: Logistic Regression, Random Forest, GBT, etc.
    """
    
    def __init__(self, feature_cols, label_col='churned', test_size=0.2, seed=42):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.test_size = test_size
        self.seed = seed
        self.model = None
        self.train_df = None
        self.test_df = None
        self.predictions = None
        
    def prepare_data(self, df):
        """Split data into train/test"""
        self.train_df, self.test_df = df.randomSplit([1-self.test_size, self.test_size], seed=self.seed)
        print(f"✓ Data split: {self.train_df.count()} train, {self.test_df.count()} test")
        
    def build_pipeline(self, algorithm='logistic_regression', **kwargs):
        """
        Build ML pipeline with specified algorithm
        
        Args:
            algorithm: 'logistic_regression', 'random_forest', 'gbt'
            **kwargs: Algorithm-specific parameters
        """
        # Feature engineering stages (common to all algorithms)
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        
        # Select algorithm
        if algorithm == 'logistic_regression':
            classifier = LogisticRegression(
                featuresCol="scaled_features",
                labelCol=self.label_col,
                maxIter=kwargs.get('max_iter', 100),
                regParam=kwargs.get('reg_param', 0.1),
                elasticNetParam=kwargs.get('elastic_net', 0.0)
            )
        # elif algorithm == 'random_forest':
        #     from sklearn.ensemble import RandomForestClassifier
        #     classifier = RandomForestClassifier(
        #         n_estimators=kwargs.get('num_trees', 100),
        #         max_depth=kwargs.get('max_depth', 5),
        #         random_state=self.seed,
        #         n_jobs=-1
        #     )
            
        # elif algorithm == 'gbt':
        #     from sklearn.ensemble import GradientBoostingClassifier
        #     classifier = GradientBoostingClassifier(
        #         n_estimators=kwargs.get('max_iter', 100),
        #         max_depth=kwargs.get('max_depth', 5),
        #         learning_rate=0.1,
        #         random_state=self.seed
        #     )    
        elif algorithm == 'random_forest':
            classifier = RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol=self.label_col,
                numTrees=kwargs.get('num_trees', 100),
                maxDepth=kwargs.get('max_depth', 5),
                seed=self.seed
            )
            
        elif algorithm == 'gbt':
            classifier = GBTClassifier(
                featuresCol="scaled_features",
                labelCol=self.label_col,
                maxIter=kwargs.get('max_iter', 100),
                maxDepth=kwargs.get('max_depth', 5),
                seed=self.seed
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Build pipeline
        pipeline = Pipeline(stages=[assembler, scaler, classifier])
        return pipeline, algorithm
    
    def train(self, algorithm='logistic_regression', **kwargs):
        """Train the model"""
        if self.train_df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print(f"\nTraining {algorithm} model...")
        pipeline, algo_name = self.build_pipeline(algorithm, **kwargs)
        
        self.model = pipeline.fit(self.train_df)
        self.algorithm = algo_name
        
        print(f"✓ Model trained successfully")
        return self.model
    
    def predict(self):
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Making predictions...")
        self.predictions = self.model.transform(self.test_df)
        print("✓ Predictions complete")
        
        return self.predictions
    
    def get_feature_importance(self):
        """Extract feature importance (algorithm-dependent)"""
        if self.algorithm == 'logistic_regression':
            lr_model = self.model.stages[-1]
            coefficients = lr_model.coefficients.toArray()
            importance = pd.DataFrame({
                'Feature': self.feature_cols,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=False)
            
        elif self.algorithm in ['random_forest', 'gbt']:
            tree_model = self.model.stages[-1]
            importance = pd.DataFrame({
                'Feature': self.feature_cols,
                'Importance': tree_model.featureImportances.toArray()
            }).sort_values('Importance', ascending=False)
        
        return importance

# COMMAND ----------

