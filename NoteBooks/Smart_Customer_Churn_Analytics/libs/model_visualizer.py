# Databricks notebook source
# ===========================================================================
# FILE: model_visualizer.py
# ===========================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ModelVisualizer:
    """
    Generate standalone visualizations.
    Can be used independently or with MLflow.
    """
    
    def __init__(self, evaluator, feature_cols, output_dir='/tmp'):
        self.evaluator = evaluator
        self.feature_cols = feature_cols
        self.output_dir = output_dir
        self.metrics = evaluator.metrics
        self.predictions_pd = evaluator.predictions_pd
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_roc_curve(self, save=True, show=True):
        """Generate ROC curve"""
        roc_data = self.metrics['roc_curve']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {self.metrics["auc_roc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/roc_curve.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_precision_recall_curve(self, save=True, show=True):
        """Generate Precision-Recall curve"""
        pr_data = self.metrics['pr_curve']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(pr_data['recall'], pr_data['precision'], color='blue', lw=2, 
                label=f'PR curve (AUC = {self.metrics["auc_pr"]:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/precision_recall_curve.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_confusion_matrix(self, save=True, show=True):
        """Generate confusion matrix heatmap"""
        cm = self.metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'],
                    annot_kws={"size": 14, "weight": "bold"},
                    ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/confusion_matrix.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_feature_importance(self, feature_importance_df, save=True, show=True):
        """Generate feature importance chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Coefficient' in feature_importance_df.columns:
            # Logistic Regression
            colors = ['green' if x > 0 else 'red' for x in feature_importance_df['Coefficient']]
            ax.barh(feature_importance_df['Feature'], feature_importance_df['Coefficient'], 
                   color=colors, alpha=0.7)
            ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        else:
            # Tree-based models
            ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
                   color='steelblue', alpha=0.7)
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/feature_importance.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_threshold_analysis(self, save=True, show=True):
        """Generate threshold analysis chart"""
        threshold_df = self.metrics['threshold_analysis']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', 
               label='Precision', linewidth=2, markersize=6)
        ax.plot(threshold_df['threshold'], threshold_df['recall'], 's-', 
               label='Recall', linewidth=2, markersize=6)
        ax.plot(threshold_df['threshold'], threshold_df['f1_score'], '^-', 
               label='F1-Score', linewidth=2, markersize=6)
        ax.plot(threshold_df['threshold'], threshold_df['accuracy'], 'd-', 
               label='Accuracy', linewidth=2, markersize=6)
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                  label='Default Threshold (0.5)')
        ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title('Performance Across Thresholds', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.05, 0.95])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/threshold_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_feature_distributions(self, save=True, show=True):
        """Generate feature distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(self.feature_cols):
            ax = axes[idx]
            ax.hist(self.predictions_pd[self.predictions_pd['churned']==0][feature], 
                   bins=30, alpha=0.5, label='Not Churned', color='blue', edgecolor='black')
            ax.hist(self.predictions_pd[self.predictions_pd['churned']==1][feature], 
                   bins=30, alpha=0.5, label='Churned', color='red', edgecolor='black')
            
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature.replace("_", " ").title()} Distribution', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/feature_distributions.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def plot_metrics_dashboard(self, save=True, show=True):
        """Generate comprehensive metrics dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        metrics = [
            ('AUC-ROC', self.metrics['auc_roc'], 'green'),
            ('Accuracy', self.metrics['accuracy'], 'blue'),
            ('Precision', self.metrics['precision'], 'orange'),
            ('Recall', self.metrics['recall'], 'purple'),
            ('F1-Score', self.metrics['f1_score'], 'red'),
            ('AUC-PR', self.metrics['auc_pr'], 'brown')
        ]
        
        for i, (metric_name, metric_value, color) in enumerate(metrics):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.text(0.5, 0.6, f'{metric_value:.4f}', 
                   ha='center', va='center', fontsize=36, fontweight='bold', color=color)
            ax.text(0.5, 0.3, metric_name, 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                                edgecolor=color, linewidth=4, transform=ax.transAxes)
            ax.add_patch(rect)
        
        # Confusion matrix in bottom row
        ax_cm = fig.add_subplot(gs[2, :])
        cm = self.metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()

        cm_display = np.array([
            [f'TN: {tn}\n({tn/total*100:.1f}%)', f'FP: {fp}\n({fp/total*100:.1f}%)'],
            [f'FN: {fn}\n({fn/total*100:.1f}%)', f'TP: {tp}\n({tp/total*100:.1f}%)']
        ])

        im = ax_cm.imshow([[tn, fp], [fn, tp]], cmap='Blues', alpha=0.6)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Predicted Not Churned', 'Predicted Churned'], 
                            fontsize=11, fontweight='bold')
        ax_cm.set_yticklabels(['Actually Not Churned', 'Actually Churned'], 
                            fontsize=11, fontweight='bold')
        ax_cm.set_title('Confusion Matrix Details', fontsize=14, fontweight='bold', pad=20)
        
        for i in range(2):
            for j in range(2):
                text = ax_cm.text(j, i, cm_display[i, j],
                                ha="center", va="center", color="black", 
                                fontsize=12, fontweight='bold')
        
        if save:
            filepath = f'{self.output_dir}/metrics_dashboard.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath if save else None
    
    def generate_all_visualizations(self, feature_importance_df=None, save=True, show=False):
        """Generate all visualizations at once"""
        print("\nGenerating all visualizations...")
        
        filepaths = {}
        filepaths['roc_curve'] = self.plot_roc_curve(save=save, show=show)
        filepaths['pr_curve'] = self.plot_precision_recall_curve(save=save, show=show)
        filepaths['confusion_matrix'] = self.plot_confusion_matrix(save=save, show=show)
        filepaths['threshold_analysis'] = self.plot_threshold_analysis(save=save, show=show)
        filepaths['feature_distributions'] = self.plot_feature_distributions(save=save, show=show)
        filepaths['metrics_dashboard'] = self.plot_metrics_dashboard(save=save, show=show)
        
        if feature_importance_df is not None:
            filepaths['feature_importance'] = self.plot_feature_importance(
                feature_importance_df, save=save, show=show
            )
        
        print("✓ All visualizations generated")
        return filepaths

# COMMAND ----------

