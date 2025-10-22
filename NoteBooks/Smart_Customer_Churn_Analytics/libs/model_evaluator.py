# Databricks notebook source
# ===========================================================================
# FILE: model_evaluator.py
# ===========================================================================

import pandas as pd
import numpy as np
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

class ModelEvaluator:
    """
    Evaluate ML models and export metrics to various formats.
    Works with any classification model.
    """
    
    def __init__(self, predictions_spark_df, label_col='churned'):
        self.predictions_spark = predictions_spark_df
        self.label_col = label_col
        
        # Convert to pandas for sklearn metrics
        self.predictions_pd = predictions_spark_df.select(
            "customer_id", label_col, "prediction", "probability",
            "recency", "frequency", "monetary", "tenure_days"
        ).toPandas()
        
        # Extract probabilities
        self.predictions_pd['churn_probability'] = self.predictions_pd['probability'].apply(
            lambda x: float(x[1])
        )
        
        self.y_true = self.predictions_pd[label_col].values
        self.y_pred = self.predictions_pd['prediction'].values
        self.y_prob = self.predictions_pd['churn_probability'].values
        
        self.metrics = {}
        
    def calculate_all_metrics(self):
        """Calculate comprehensive metrics"""
        # Spark-based metrics
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol=self.label_col, 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderROC"
        )
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol=self.label_col, 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderPR"
        )
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col, 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        
        self.metrics['auc_roc'] = evaluator_auc.evaluate(self.predictions_spark)
        self.metrics['auc_pr'] = evaluator_pr.evaluate(self.predictions_spark)
        self.metrics['accuracy'] = accuracy_evaluator.evaluate(self.predictions_spark)
        
        # Confusion matrix metrics
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics['confusion_matrix'] = cm
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)
        
        self.metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.metrics['f1_score'] = (
            2 * self.metrics['precision'] * self.metrics['recall'] / 
            (self.metrics['precision'] + self.metrics['recall'])
            if (self.metrics['precision'] + self.metrics['recall']) > 0 else 0
        )
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_prob)
        self.metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
        
        # Precision-Recall curve data
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(self.y_true, self.y_prob)
        self.metrics['pr_curve'] = {
            'precision': precision_curve, 
            'recall': recall_curve, 
            'thresholds': pr_thresholds
        }
        
        # Threshold analysis
        self.metrics['threshold_analysis'] = self._analyze_thresholds()
        
        print("✓ All metrics calculated")
        return self.metrics
    
    def _analyze_thresholds(self):
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_threshold = (self.y_prob >= threshold).astype(int)
            cm_temp = confusion_matrix(self.y_true, y_pred_threshold)
            tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
            
            prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
            
            results.append({
                'threshold': threshold,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'accuracy': acc
            })
        
        return pd.DataFrame(results)
    
    def get_metrics_summary(self):
        """Get metrics as dictionary (for export to Power BI, Qlik, etc.)"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        summary = {
            'auc_roc': self.metrics['auc_roc'],
            'auc_pr': self.metrics['auc_pr'],
            'accuracy': self.metrics['accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1_score': self.metrics['f1_score'],
            'specificity': self.metrics['specificity'],
            'true_positives': self.metrics['true_positives'],
            'true_negatives': self.metrics['true_negatives'],
            'false_positives': self.metrics['false_positives'],
            'false_negatives': self.metrics['false_negatives']
        }
        
        return summary
    
    def export_to_dataframe(self):
        """Export predictions with probabilities (for Power BI, Qlik)"""
        return self.predictions_pd
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=['Not Churned', 'Churned'],
            output_dict=True
        )
    
    def print_summary(self):
        """Print formatted metrics summary"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE METRICS")
        print("="*80)
        print(f"AUC-ROC:          {self.metrics['auc_roc']:.4f}")
        print(f"AUC-PR:           {self.metrics['auc_pr']:.4f}")
        print(f"Accuracy:         {self.metrics['accuracy']:.4f}")
        print(f"Precision:        {self.metrics['precision']:.4f}")
        print(f"Recall:           {self.metrics['recall']:.4f}")
        print(f"F1-Score:         {self.metrics['f1_score']:.4f}")
        print(f"Specificity:      {self.metrics['specificity']:.4f}")
        print("="*80)
        print("\nConfusion Matrix:")
        print(f"True Negatives:   {self.metrics['true_negatives']}")
        print(f"False Positives:  {self.metrics['false_positives']}")
        print(f"False Negatives:  {self.metrics['false_negatives']}")
        print(f"True Positives:   {self.metrics['true_positives']}")
        print("="*80)

# COMMAND ----------

