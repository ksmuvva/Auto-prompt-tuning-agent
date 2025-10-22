"""
Comparative Analyzer
Compare multiple prompts, models, and strategies side-by-side with tabulated results
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """
    Compares multiple prompts, models, or strategies
    Generates comparison tables and recommendations
    """

    def __init__(self):
        self.comparison_history = []

    def compare_prompts(
        self,
        prompt_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple prompt performances
        
        Args:
            prompt_results: List of dictionaries with prompt names and metrics
        
        Returns:
            DataFrame with comparative analysis
        """
        if not prompt_results:
            logger.warning("No prompt results to compare")
            return pd.DataFrame()

        comparison_data = []
        
        for result in prompt_results:
            comparison_data.append({
                'Prompt': result.get('prompt_name', 'Unknown'),
                'Precision': f"{result.get('precision', 0):.2%}",
                'Accuracy': f"{result.get('accuracy', 0):.2%}",
                'Recall': f"{result.get('recall', 0):.2%}",
                'F1 Score': f"{result.get('f1_score', 0):.2%}",
                'Composite Score': f"{result.get('composite_score', 0):.2%}",
                'Latency (s)': f"{result.get('latency', 0):.2f}",
                'Tokens': result.get('tokens_used', 0),
                'Meets 98% Target': '✓' if result.get('precision', 0) >= 0.98 and result.get('accuracy', 0) >= 0.98 else '✗'
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by composite score (descending)
        if 'Composite Score' in df.columns:
            df = df.sort_values('Composite Score', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
        
        logger.info(f"Compared {len(prompt_results)} prompts")
        return df

    def compare_models(
        self,
        model_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple LLM model performances
        
        Args:
            model_results: List of dictionaries with model names and metrics
        
        Returns:
            DataFrame with model comparison
        """
        if not model_results:
            logger.warning("No model results to compare")
            return pd.DataFrame()

        comparison_data = []
        
        for result in model_results:
            comparison_data.append({
                'Model': result.get('model_name', 'Unknown'),
                'Provider': result.get('provider', 'Unknown'),
                'Precision': f"{result.get('precision', 0):.2%}",
                'Accuracy': f"{result.get('accuracy', 0):.2%}",
                'F1 Score': f"{result.get('f1_score', 0):.2%}",
                'Avg Latency (s)': f"{result.get('avg_latency', 0):.2f}",
                'Avg Tokens': result.get('avg_tokens', 0),
                'Est. Cost ($)': f"${result.get('estimated_cost', 0):.4f}",
                'Meets Target': '✓' if result.get('precision', 0) >= 0.98 and result.get('accuracy', 0) >= 0.98 else '✗',
                'Recommended': result.get('recommended', '')
            })
        
        df = pd.DataFrame(comparison_data)
        
        logger.info(f"Compared {len(model_results)} models")
        return df

    def compare_strategies(
        self,
        strategy_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare different analysis strategies (template vs dynamic vs hybrid)
        
        Args:
            strategy_results: List of strategy results
        
        Returns:
            DataFrame with strategy comparison
        """
        if not strategy_results:
            logger.warning("No strategy results to compare")
            return pd.DataFrame()

        comparison_data = []
        
        for result in strategy_results:
            comparison_data.append({
                'Strategy': result.get('strategy_name', 'Unknown'),
                'Precision': f"{result.get('precision', 0):.2%}",
                'Accuracy': f"{result.get('accuracy', 0):.2%}",
                'F1 Score': f"{result.get('f1_score', 0):.2%}",
                'Execution Time (s)': f"{result.get('execution_time', 0):.1f}",
                'Iterations': result.get('iterations', 1),
                'Final Prompt Quality': result.get('prompt_quality', 'N/A'),
                'Target Met': '✓' if result.get('target_met', False) else '✗'
            })
        
        df = pd.DataFrame(comparison_data)
        
        logger.info(f"Compared {len(strategy_results)} strategies")
        return df

    def generate_comparison_table(
        self,
        comparison_type: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate ASCII comparison table for CLI display
        
        Args:
            comparison_type: 'prompts', 'models', or 'strategies'
            results: List of result dictionaries
        
        Returns:
            Formatted ASCII table string
        """
        if comparison_type == 'prompts':
            df = self.compare_prompts(results)
        elif comparison_type == 'models':
            df = self.compare_models(results)
        elif comparison_type == 'strategies':
            df = self.compare_strategies(results)
        else:
            logger.error(f"Unknown comparison type: {comparison_type}")
            return ""

        if df.empty:
            return "No data to display"

        # Convert DataFrame to ASCII table
        table = self._dataframe_to_ascii_table(df)
        
        return table

    def _dataframe_to_ascii_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to ASCII table"""
        if df.empty:
            return "Empty table"
        
        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = max(
                len(str(col)),
                df[col].astype(str).str.len().max()
            ) + 2  # Padding
        
        # Build table
        lines = []
        
        # Header separator
        lines.append("┌" + "┬".join("─" * col_widths[col] for col in df.columns) + "┐")
        
        # Header row
        header = "│" + "│".join(f" {col:^{col_widths[col]-2}} " for col in df.columns) + "│"
        lines.append(header)
        
        # Header-data separator
        lines.append("├" + "┼".join("─" * col_widths[col] for col in df.columns) + "┤")
        
        # Data rows
        for _, row in df.iterrows():
            row_str = "│" + "│".join(
                f" {str(row[col]):^{col_widths[col]-2}} " for col in df.columns
            ) + "│"
            lines.append(row_str)
        
        # Bottom separator
        lines.append("└" + "┴".join("─" * col_widths[col] for col in df.columns) + "┘")
        
        return "\n".join(lines)

    def recommend_best_option(
        self,
        comparison_type: str,
        results: List[Dict[str, Any]],
        criteria: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Recommend the best option based on criteria
        
        Args:
            comparison_type: 'prompts', 'models', or 'strategies'
            results: List of result dictionaries
            criteria: 'performance' (best metrics), 'speed' (fastest), 
                     'cost' (cheapest), or 'balanced' (best overall)
        
        Returns:
            Recommendation dictionary
        """
        if not results:
            return {'error': 'No results to analyze'}

        if criteria == 'performance':
            # Best precision + accuracy
            best = max(results, key=lambda x: x.get('precision', 0) + x.get('accuracy', 0))
            reason = "Highest combined precision and accuracy"
        
        elif criteria == 'speed':
            # Fastest execution
            best = min(results, key=lambda x: x.get('latency', float('inf')))
            reason = "Fastest execution time"
        
        elif criteria == 'cost':
            # Lowest cost
            best = min(results, key=lambda x: x.get('estimated_cost', float('inf')))
            reason = "Lowest estimated cost"
        
        else:  # balanced
            # Weighted score: 50% performance, 30% speed, 20% cost
            def balanced_score(r):
                perf = (r.get('precision', 0) + r.get('accuracy', 0)) / 2
                speed = 1 / (r.get('latency', 1) + 0.1)  # Inverse for speed
                cost = 1 / (r.get('estimated_cost', 0.001) + 0.001)  # Inverse for cost
                return (perf * 0.5) + (speed * 0.3) + (cost * 0.2)
            
            best = max(results, key=balanced_score)
            reason = "Best balance of performance, speed, and cost"

        recommendation = {
            'recommended': best,
            'reason': reason,
            'criteria': criteria,
            'name': best.get(f'{comparison_type[:-1]}_name', best.get('name', 'Unknown')),
            'key_metrics': {
                'precision': best.get('precision', 0),
                'accuracy': best.get('accuracy', 0),
                'f1_score': best.get('f1_score', 0)
            },
            'meets_target': best.get('precision', 0) >= 0.98 and best.get('accuracy', 0) >= 0.98
        }

        logger.info(f"Recommended: {recommendation['name']} ({reason})")
        
        return recommendation

    def generate_comprehensive_report(
        self,
        prompt_results: Optional[List[Dict[str, Any]]] = None,
        model_results: Optional[List[Dict[str, Any]]] = None,
        strategy_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate comprehensive comparison report
        
        Returns:
            Formatted report string
        """
        report_lines = []
        
        report_lines.append("="*70)
        report_lines.append("COMPREHENSIVE COMPARATIVE ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Prompt comparison
        if prompt_results:
            report_lines.append("PROMPT COMPARISON")
            report_lines.append("-"*70)
            prompt_table = self.generate_comparison_table('prompts', prompt_results)
            report_lines.append(prompt_table)
            
            best_prompt = self.recommend_best_option('prompts', prompt_results, 'performance')
            report_lines.append(f"\n✓ Recommended Prompt: {best_prompt['name']}")
            report_lines.append(f"  Reason: {best_prompt['reason']}")
            report_lines.append(f"  Precision: {best_prompt['key_metrics']['precision']:.2%}")
            report_lines.append(f"  Accuracy: {best_prompt['key_metrics']['accuracy']:.2%}")
            report_lines.append("")

        # Model comparison
        if model_results:
            report_lines.append("MODEL COMPARISON")
            report_lines.append("-"*70)
            model_table = self.generate_comparison_table('models', model_results)
            report_lines.append(model_table)
            
            best_model = self.recommend_best_option('models', model_results, 'balanced')
            report_lines.append(f"\n✓ Recommended Model: {best_model['name']}")
            report_lines.append(f"  Reason: {best_model['reason']}")
            report_lines.append(f"  Precision: {best_model['key_metrics']['precision']:.2%}")
            report_lines.append(f"  Accuracy: {best_model['key_metrics']['accuracy']:.2%}")
            report_lines.append("")

        # Strategy comparison
        if strategy_results:
            report_lines.append("STRATEGY COMPARISON")
            report_lines.append("-"*70)
            strategy_table = self.generate_comparison_table('strategies', strategy_results)
            report_lines.append(strategy_table)
            
            best_strategy = self.recommend_best_option('strategies', strategy_results, 'performance')
            report_lines.append(f"\n✓ Recommended Strategy: {best_strategy['name']}")
            report_lines.append(f"  Reason: {best_strategy['reason']}")
            report_lines.append("")

        report_lines.append("="*70)
        
        report = "\n".join(report_lines)
        
        # Store in history
        self.comparison_history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt_count': len(prompt_results) if prompt_results else 0,
            'model_count': len(model_results) if model_results else 0,
            'strategy_count': len(strategy_results) if strategy_results else 0,
            'report': report
        })
        
        return report

    def export_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_file: str,
        format: str = 'csv'
    ):
        """
        Export comparison to file
        
        Args:
            comparison_df: DataFrame to export
            output_file: Output file path
            format: 'csv', 'json', or 'excel'
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            comparison_df.to_csv(output_path, index=False)
        elif format == 'json':
            comparison_df.to_json(output_path, orient='records', indent=2)
        elif format == 'excel':
            comparison_df.to_excel(output_path, index=False)
        else:
            logger.error(f"Unsupported format: {format}")
            return

        logger.info(f"Exported comparison to {output_path}")

    def export_history(self, filepath: str):
        """Export comparison history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.comparison_history, f, indent=2, default=str)
        
        logger.info(f"Exported comparison history to {filepath}")
