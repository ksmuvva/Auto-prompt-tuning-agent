"""
Data Processor Module
Handles CSV bank transaction data processing and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionDataProcessor:
    """Process and validate bank transaction CSV files"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transactions = []
        self.summary_stats = {}

    def load_csv_files(self, pattern: str = "*.csv") -> List[pd.DataFrame]:
        """Load all CSV files matching the pattern"""
        csv_files = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(csv_files)} CSV files")

        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {csv_file.name}: {len(df)} transactions")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        return dataframes

    def merge_transactions(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all transaction dataframes"""
        if not dataframes:
            logger.warning("No dataframes to merge")
            return pd.DataFrame()

        merged = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Merged {len(merged)} total transactions")

        # Standardize column names
        merged.columns = merged.columns.str.lower().str.strip()

        return merged

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate transaction data quality"""
        validation_report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'date_range': None,
            'currency_types': [],
            'amount_stats': {}
        }

        # Check for required columns
        required_cols = ['amount', 'date', 'description']
        available_cols = [col for col in required_cols if col in df.columns]

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            validation_report['date_range'] = {
                'min': str(df['date'].min()),
                'max': str(df['date'].max())
            }

        if 'amount' in df.columns:
            validation_report['amount_stats'] = {
                'min': float(df['amount'].min()),
                'max': float(df['amount'].max()),
                'mean': float(df['amount'].mean()),
                'median': float(df['amount'].median()),
                'std': float(df['amount'].std())
            }

        if 'currency' in df.columns:
            validation_report['currency_types'] = df['currency'].unique().tolist()

        return validation_report

    def filter_transactions_above_threshold(
        self,
        df: pd.DataFrame,
        threshold: float = 250.0,
        currency: str = 'GBP'
    ) -> pd.DataFrame:
        """Filter transactions above specified threshold"""
        if 'amount' not in df.columns:
            logger.error("'amount' column not found")
            return pd.DataFrame()

        # Filter by currency if column exists
        if 'currency' in df.columns:
            df_filtered = df[
                (df['amount'] > threshold) &
                (df['currency'] == currency)
            ]
        else:
            df_filtered = df[df['amount'] > threshold]

        logger.info(f"Found {len(df_filtered)} transactions above {threshold} {currency}")
        return df_filtered

    def prepare_for_llm_analysis(
        self,
        df: pd.DataFrame,
        max_rows: int = 1000
    ) -> str:
        """Prepare transaction data as formatted text for LLM analysis"""
        if len(df) > max_rows:
            logger.warning(f"Sampling {max_rows} from {len(df)} transactions")
            df = df.sample(n=max_rows, random_state=42)

        # Create a concise representation
        data_summary = f"""
TRANSACTION DATA SUMMARY
========================
Total Transactions: {len(df)}
Date Range: {df['date'].min()} to {df['date'].max() if 'date' in df.columns else 'N/A'}

SAMPLE TRANSACTIONS:
"""
        # Add sample transactions
        for idx, row in df.head(50).iterrows():
            data_summary += f"\n{idx + 1}. "
            data_summary += " | ".join([f"{col}: {row[col]}" for col in df.columns])

        if len(df) > 50:
            data_summary += f"\n\n... and {len(df) - 50} more transactions"

        return data_summary

    def get_ground_truth_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 250.0
    ) -> Dict[str, Any]:
        """Generate ground truth for evaluation"""
        ground_truth = {
            'high_value_transactions': [],
            'count_above_threshold': 0,
            'total_amount_above_threshold': 0.0
        }

        if 'amount' in df.columns:
            high_value = df[df['amount'] > threshold]
            ground_truth['count_above_threshold'] = len(high_value)
            ground_truth['total_amount_above_threshold'] = float(high_value['amount'].sum())

            # Store transaction IDs or indices
            if 'transaction_id' in df.columns:
                ground_truth['high_value_transactions'] = high_value['transaction_id'].tolist()
            else:
                ground_truth['high_value_transactions'] = high_value.index.tolist()

        return ground_truth

    def detect_statistical_anomalies(
        self,
        df: pd.DataFrame,
        z_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect anomalies using statistical methods (Z-score)"""
        if 'amount' not in df.columns:
            return pd.DataFrame()

        df = df.copy()

        # Calculate Z-score
        mean = df['amount'].mean()
        std = df['amount'].std()
        df['z_score'] = (df['amount'] - mean) / std

        # Flag anomalies
        df['is_anomaly'] = abs(df['z_score']) > z_threshold

        anomalies = df[df['is_anomaly']]
        logger.info(f"Detected {len(anomalies)} statistical anomalies")

        return anomalies

    def process_all(self) -> Dict[str, Any]:
        """Complete data processing pipeline"""
        logger.info("Starting complete data processing pipeline")

        # Load all CSV files
        dataframes = self.load_csv_files()

        if not dataframes:
            logger.error("No data files found")
            return {}

        # Merge all transactions
        df = self.merge_transactions(dataframes)

        # Validate data
        validation = self.validate_data(df)

        # Filter high-value transactions
        high_value = self.filter_transactions_above_threshold(df, threshold=250.0)

        # Detect anomalies
        anomalies = self.detect_statistical_anomalies(df)

        # Prepare for LLM
        llm_data = self.prepare_for_llm_analysis(df)

        # Ground truth
        ground_truth = self.get_ground_truth_labels(df)

        return {
            'full_data': df,
            'validation_report': validation,
            'high_value_transactions': high_value,
            'statistical_anomalies': anomalies,
            'llm_formatted_data': llm_data,
            'ground_truth': ground_truth
        }
