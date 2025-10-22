# Databricks notebook source
# ===========================================================================
# FILE: report_generator.py
# ===========================================================================

import json
import pandas as pd
from datetime import datetime

# COMMAND ----------



class ReportGenerator:
    """
    Export results to multiple formats:
    - JSON (for GenAI)
    - CSV (for Power BI, Qlik)
    - HTML (for viewing)
    - Parquet (for data lake)
    """
    
    def __init__(self, evaluator, trainer, output_dir='/tmp'):
        self.evaluator = evaluator
        self.trainer = trainer
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def export_to_json(self, include_predictions=False):
        """Export metrics to JSON (perfect for GenAI input)"""
        metrics_summary = self.evaluator.get_metrics_summary()
        classification_report = self.evaluator.get_classification_report()
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'algorithm': self.trainer.algorithm,
                'feature_columns': self.trainer.feature_cols,
                'test_size': self.trainer.test_size
            },
            'metrics': metrics_summary,
            'classification_report': classification_report,
            'confusion_matrix': {
                'true_negatives': int(metrics_summary['true_negatives']),
                'false_positives': int(metrics_summary['false_positives']),
                'false_negatives': int(metrics_summary['false_negatives']),
                'true_positives': int(metrics_summary['true_positives'])
            }
        }
        
        if include_predictions:
            predictions_sample = self.evaluator.predictions_pd.head(100).to_dict('records')
            export_data['predictions_sample'] = predictions_sample
        
        filepath = f'{self.output_dir}/model_results_{self.timestamp}.json'
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ JSON export saved: {filepath}")
        return filepath
    
    def export_predictions_to_csv(self):
        """Export predictions to CSV (for Power BI, Qlik, Excel)"""
        predictions_df = self.evaluator.export_to_dataframe()
        
        filepath = f'{self.output_dir}/predictions_{self.timestamp}.csv'
        predictions_df.to_csv(filepath, index=False)
        
        print(f"✓ Predictions CSV saved: {filepath}")
        return filepath
    
    def export_metrics_to_csv(self):
        """Export metrics to CSV (for dashboards)"""
        metrics_summary = self.evaluator.get_metrics_summary()
        
        metrics_df = pd.DataFrame([metrics_summary])
        metrics_df['timestamp'] = datetime.now()
        metrics_df['algorithm'] = self.trainer.algorithm
        
        filepath = f'{self.output_dir}/metrics_{self.timestamp}.csv'
        metrics_df.to_csv(filepath, index=False)
        
        print(f"✓ Metrics CSV saved: {filepath}")
        return filepath
    
    def export_threshold_analysis_to_csv(self):
        """Export threshold analysis (for optimization)"""
        threshold_df = self.evaluator.metrics['threshold_analysis']
        
        filepath = f'{self.output_dir}/threshold_analysis_{self.timestamp}.csv'
        threshold_df.to_csv(filepath, index=False)
        
        print(f"✓ Threshold analysis CSV saved: {filepath}")
        return filepath
    
    def export_to_parquet(self):
        """Export predictions to Parquet (for data lake)"""
        predictions_df = self.evaluator.export_to_dataframe()
        
        filepath = f'{self.output_dir}/predictions_{self.timestamp}.parquet'
        predictions_df.to_parquet(filepath, index=False)
        
        print(f"✓ Parquet export saved: {filepath}")
        return filepath
    
    def generate_html_report(self, visualization_paths=None):
        """Generate comprehensive HTML report"""
        metrics_summary = self.evaluator.get_metrics_summary()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Model Report - {self.timestamp}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 36px;
                    font-weight: 700;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .content {{
                    padding: 40px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                }}
                .metric-value {{
                    font-size: 42px;
                    font-weight: bold;
                    color: #667eea;
                    margin: 10px 0;
                }}
                .metric-name {{
                    font-size: 14px;
                    color: #555;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .section {{
                    margin: 40px 0;
                }}
                .section h2 {{
                    color: #333;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .viz-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .viz-container img {{
                    max-width: 100%;
                    border-radius: 10px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    margin: 10px 0;
                }}
                .confusion-matrix {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 15px;
                    margin: 20px 0;
                }}
                .cm-cell {{
                    background: #f5f7fa;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .cm-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .cm-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
                .footer {{
                    background: #f5f7fa;
                    padding: 20px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Customer Churn Prediction Model Report</h1>
                    <p>Algorithm: {self.trainer.algorithm.upper()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="content">
                    <div class="section">
                        <h2>📊 Performance Metrics</h2>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-name">AUC-ROC</div>
                                <div class="metric-value">{metrics_summary['auc_roc']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Accuracy</div>
                                <div class="metric-value">{metrics_summary['accuracy']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Precision</div>
                                <div class="metric-value">{metrics_summary['precision']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Recall</div>
                                <div class="metric-value">{metrics_summary['recall']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">F1-Score</div>
                                <div class="metric-value">{metrics_summary['f1_score']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Specificity</div>
                                <div class="metric-value">{metrics_summary['specificity']:.4f}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🎯 Confusion Matrix</h2>
                        <div class="confusion-matrix">
                            <div class="cm-cell">
                                <div class="cm-value">{metrics_summary['true_negatives']}</div>
                                <div class="cm-label">True Negatives</div>
                            </div>
                            <div class="cm-cell">
                                <div class="cm-value">{metrics_summary['false_positives']}</div>
                                <div class="cm-label">False Positives</div>
                            </div>
                            <div class="cm-cell">
                                <div class="cm-value">{metrics_summary['false_negatives']}</div>
                                <div class="cm-label">False Negatives</div>
                            </div>
                            <div class="cm-cell">
                                <div class="cm-value">{metrics_summary['true_positives']}</div>
                                <div class="cm-label">True Positives</div>
                            </div>
                        </div>
                    </div>
        """
        
        # Add visualizations if provided
        if visualization_paths:
            html_content += """
                    <div class="section">
                        <h2>📈 Visualizations</h2>
            """
            
            for viz_name, viz_path in visualization_paths.items():
                if viz_path:
                    html_content += f"""
                        <div class="viz-container">
                            <h3>{viz_name.replace('_', ' ').title()}</h3>
                            <img src="{viz_path}" alt="{viz_name}">
                        </div>
                    """
            
            html_content += "</div>"
        
        html_content += """
                </div>
                
                <div class="footer">
                    <p>Generated by ML Pipeline | Powered by Databricks & MLflow</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        filepath = f'{self.output_dir}/model_report_{self.timestamp}.html'
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved: {filepath}")
        return filepath
    
    def generate_genai_prompt(self):
        """Generate a prompt for GenAI to explain results"""
        metrics_summary = self.evaluator.get_metrics_summary()
        classification_report = self.evaluator.get_classification_report()
        
        prompt = f"""
Analyze the following machine learning model results and provide a comprehensive explanation:

**Model Information:**
- Algorithm: {self.trainer.algorithm}
- Features: {', '.join(self.trainer.feature_cols)}
- Test Set Size: {len(self.evaluator.predictions_pd)} customers

**Performance Metrics:**
- AUC-ROC: {metrics_summary['auc_roc']:.4f}
- Accuracy: {metrics_summary['accuracy']:.4f}
- Precision: {metrics_summary['precision']:.4f}
- Recall: {metrics_summary['recall']:.4f}
- F1-Score: {metrics_summary['f1_score']:.4f}
- Specificity: {metrics_summary['specificity']:.4f}

**Confusion Matrix:**
- True Positives: {metrics_summary['true_positives']} (correctly identified churners)
- True Negatives: {metrics_summary['true_negatives']} (correctly identified non-churners)
- False Positives: {metrics_summary['false_positives']} (false alarms)
- False Negatives: {metrics_summary['false_negatives']} (missed churners)

Please provide:
1. An executive summary of model performance
2. Interpretation of key metrics (what do they mean for the business?)
3. Strengths and weaknesses of the model
4. Recommendations for improvement
5. Business implications and next steps
"""
        
        filepath = f'{self.output_dir}/genai_prompt_{self.timestamp}.txt'
        with open(filepath, 'w') as f:
            f.write(prompt)
        
        print(f"✓ GenAI prompt saved: {filepath}")
        return prompt, filepath
    
    def export_all(self, visualization_paths=None):
        """Export to all formats"""
        print("\n" + "="*80)
        print("EXPORTING RESULTS TO MULTIPLE FORMATS")
        print("="*80)
        
        exports = {
            'json': self.export_to_json(include_predictions=True),
            'predictions_csv': self.export_predictions_to_csv(),
            'metrics_csv': self.export_metrics_to_csv(),
            'threshold_csv': self.export_threshold_analysis_to_csv(),
            'parquet': self.export_to_parquet(),
            'html': self.generate_html_report(visualization_paths),
            'genai_prompt': self.generate_genai_prompt()[1]
        }
        
        print("="*80)
        print("✓ ALL EXPORTS COMPLETE")
        print("="*80)
        
        return exports

# COMMAND ----------

