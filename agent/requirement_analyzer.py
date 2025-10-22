"""
Requirement Analyzer
Analyzes data according to specific FW requirements (FW15-FW50)
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequirementAnalyzer:
    """
    Analyzes bank transaction data for specific FW requirements
    
    Implements:
    - FW15: High-value transactions (>£250)
    - FW20: Luxury brands & money transfers
    - FW25: Missing audit trails
    - FW30: Missing months detection
    - FW40: Light-touch fraud detection
    - FW45: Gambling transactions
    - FW50: Large debt payments
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.df = None
        
        # FW20: Luxury brand keywords
        self.luxury_keywords = [
            'gucci', 'louis vuitton', 'prada', 'chanel', 'rolex',
            'hermes', 'cartier', 'burberry', 'versace', 'dior',
            'tiffany', 'bulgari', 'armani', 'fendi', 'balenciaga'
        ]
        
        # FW20: Money transfer companies
        self.transfer_keywords = [
            'western union', 'moneygram', 'wise', 'paypal transfer',
            'revolut transfer', 'transferwise', 'worldremit', 'xoom',
            'remitly', 'azimo', 'wire transfer', 'bank transfer'
        ]
        
        # FW45: Gambling operators
        self.gambling_keywords = [
            'bet365', 'william hill', 'paddy power', 'ladbrokes',
            'betfair', 'sky bet', '888 casino', 'coral', 'betway',
            'unibet', 'casino', 'poker', 'betting'
        ]
        
        # FW50: Debt payment keywords
        self.debt_keywords = [
            'loan', 'credit card', 'mortgage', 'finance', 'repayment',
            'barclaycard', 'amex', 'visa payment', 'mastercard payment'
        ]
        
        # FW25: Missing audit trail indicators
        self.missing_audit_keywords = [
            'unknown merchant', 'cash withdrawal', 'atm', 'anonymous',
            'foreign exchange', 'unspecified'
        ]
        
        # FW40: Common bank misspellings
        self.bank_spellings = {
            'barclays': ['barcley', 'barlcays', 'barclays'],
            'hsbc': ['hscb', 'hsbc bank', 'hsbc'],
            'santander': ['santaner', 'santandar', 'santander'],
            'lloyds': ['loyds', 'llyods', 'lloyds'],
            'nationwide': ['nationwid', 'nation wide', 'nationwide']
        }

    def load_all_data(self) -> pd.DataFrame:
        """Load all CSV transaction files"""
        csv_files = list(self.data_dir.glob("transactions_*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            return pd.DataFrame()
        
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
            self.df['date'] = pd.to_datetime(self.df['date'])
            logger.info(f"Loaded {len(self.df)} total transactions")
            return self.df
        
        return pd.DataFrame()

    def analyze_fw15_high_value(self, threshold: float = 250.0) -> Dict[str, Any]:
        """
        FW15: Summarize and group all spend exceeding £250
        
        Returns:
            Dictionary with high-value transactions grouped by merchant/category
        """
        if self.df is None:
            self.load_all_data()
        
        # Filter high-value transactions
        high_value = self.df[self.df['amount'] > threshold].copy()
        
        # Group by merchant
        merchant_groups = high_value.groupby('merchant').agg({
            'amount': ['sum', 'count', 'mean'],
            'transaction_id': 'count'
        }).round(2)
        
        merchant_groups.columns = ['total_amount', 'transaction_count', 'avg_amount', 'id_count']
        merchant_groups = merchant_groups.sort_values('total_amount', ascending=False)
        
        # Group by category
        category_groups = high_value.groupby('category').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        category_groups.columns = ['total_amount', 'transaction_count', 'avg_amount']
        category_groups = category_groups.sort_values('total_amount', ascending=False)
        
        result = {
            'requirement': 'FW15',
            'description': f'Transactions exceeding £{threshold}',
            'total_count': len(high_value),
            'total_amount': float(high_value['amount'].sum()),
            'avg_amount': float(high_value['amount'].mean()),
            'max_amount': float(high_value['amount'].max()),
            'min_amount': float(high_value['amount'].min()),
            'merchant_groups': merchant_groups.to_dict('index'),
            'category_groups': category_groups.to_dict('index'),
            'transactions': high_value[['transaction_id', 'date', 'amount', 'merchant', 'category']].to_dict('records'),
            'percentage_of_total': round(len(high_value) / len(self.df) * 100, 2)
        }
        
        logger.info(f"FW15: Found {len(high_value)} transactions > £{threshold}")
        return result

    def analyze_fw20_similar_transactions(self, threshold: float = 250.0) -> Dict[str, Any]:
        """
        FW20: Identify and group similar transactions
        
        Detects:
        - Luxury brand purchases
        - Money transfer companies
        - Small transactions that collectively exceed £250/month
        """
        if self.df is None:
            self.load_all_data()
        
        # Detect luxury brands
        luxury_mask = self.df['merchant'].str.lower().str.contains(
            '|'.join(self.luxury_keywords), na=False
        )
        luxury_transactions = self.df[luxury_mask].copy()
        
        # Detect money transfers
        transfer_mask = self.df['merchant'].str.lower().str.contains(
            '|'.join(self.transfer_keywords), na=False
        )
        transfer_transactions = self.df[transfer_mask].copy()
        
        # Group small transactions by merchant and month
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        small_transactions = self.df[self.df['amount'] <= threshold].copy()
        
        monthly_merchant_totals = small_transactions.groupby(
            ['year_month', 'merchant']
        )['amount'].agg(['sum', 'count']).reset_index()
        
        # Find merchants where small transactions collectively exceed threshold
        accumulated_high = monthly_merchant_totals[
            monthly_merchant_totals['sum'] > threshold
        ]
        
        result = {
            'requirement': 'FW20',
            'description': 'Similar transactions exceeding threshold',
            'luxury_brands': {
                'count': len(luxury_transactions),
                'total_amount': float(luxury_transactions['amount'].sum()),
                'transactions': luxury_transactions[
                    ['transaction_id', 'date', 'amount', 'merchant']
                ].to_dict('records')
            },
            'money_transfers': {
                'count': len(transfer_transactions),
                'total_amount': float(transfer_transactions['amount'].sum()),
                'transactions': transfer_transactions[
                    ['transaction_id', 'date', 'amount', 'merchant']
                ].to_dict('records')
            },
            'accumulated_small_transactions': {
                'groups_exceeding_threshold': len(accumulated_high),
                'details': accumulated_high.to_dict('records')
            }
        }
        
        logger.info(f"FW20: Luxury brands={len(luxury_transactions)}, Transfers={len(transfer_transactions)}")
        return result

    def analyze_fw25_missing_audit(self) -> Dict[str, Any]:
        """
        FW25: List transfers lacking an audit trail
        
        Identifies transactions with:
        - Missing merchant information
        - Unknown merchants
        - Large cash withdrawals without details
        """
        if self.df is None:
            self.load_all_data()
        
        # Detect missing audit trail
        missing_audit = self.df[
            self.df['merchant'].str.lower().str.contains(
                '|'.join(self.missing_audit_keywords), na=False
            ) |
            self.df['merchant'].isna() |
            (self.df['notes'].str.contains('Missing audit trail', na=False))
        ].copy()
        
        # Large cash withdrawals (>£500) without proper documentation
        large_cash = self.df[
            (self.df['transaction_type'].str.contains('ATM|Cash', case=False, na=False)) &
            (self.df['amount'] > 500) &
            (self.df['notes'].isna() | (self.df['notes'] == ''))
        ].copy()
        
        # Combine unique transactions
        all_missing = pd.concat([missing_audit, large_cash]).drop_duplicates('transaction_id')
        
        result = {
            'requirement': 'FW25',
            'description': 'Transfers lacking audit trail',
            'total_count': len(all_missing),
            'unknown_merchants': len(missing_audit),
            'large_undocumented_cash': len(large_cash),
            'total_amount': float(all_missing['amount'].sum()),
            'transactions': all_missing[
                ['transaction_id', 'date', 'amount', 'merchant', 'transaction_type', 'notes']
            ].to_dict('records'),
            'risk_assessment': 'HIGH' if len(all_missing) > 50 else 'MEDIUM' if len(all_missing) > 20 else 'LOW'
        }
        
        logger.info(f"FW25: Found {len(all_missing)} transactions with missing audit trail")
        return result

    def analyze_fw30_missing_months(self, expected_months: int = 6) -> Dict[str, Any]:
        """
        FW30: Detect missing months within a 6-month bank statement sequence
        
        Returns:
            Missing months and gap analysis
        """
        if self.df is None:
            self.load_all_data()
        
        # Get date range
        min_date = self.df['date'].min()
        max_date = self.df['date'].max()
        
        # Get all unique months in the data
        actual_months = set(self.df['date'].dt.to_period('M'))
        
        # Generate expected month range
        date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        expected_months_set = set(pd.PeriodIndex(date_range, freq='M'))
        
        # Find missing months
        missing_months = expected_months_set - actual_months
        missing_months_list = sorted([str(m) for m in missing_months])
        
        # Calculate gaps
        gaps = []
        if len(missing_months) > 0:
            for month in missing_months:
                gaps.append({
                    'month': str(month),
                    'year': month.year,
                    'month_num': month.month
                })
        
        result = {
            'requirement': 'FW30',
            'description': 'Missing months in bank statement sequence',
            'date_range': {
                'start': str(min_date.date()),
                'end': str(max_date.date()),
                'span_months': len(expected_months_set)
            },
            'actual_months_count': len(actual_months),
            'expected_months_count': len(expected_months_set),
            'missing_months_count': len(missing_months),
            'missing_months': missing_months_list,
            'gap_details': gaps,
            'has_gaps': len(missing_months) > 0,
            'actual_months': sorted([str(m) for m in actual_months]),
            'is_continuous': len(missing_months) == 0
        }
        
        logger.info(f"FW30: Found {len(missing_months)} missing months: {missing_months_list}")
        return result

    def analyze_fw40_fraud_detection(self) -> Dict[str, Any]:
        """
        FW40: Light-touch fraud detection
        
        Detects:
        - Bank name misspellings
        - Calculation errors
        - Misspelled words
        - Data quality issues
        """
        if self.df is None:
            self.load_all_data()
        
        errors = {
            'misspellings': [],
            'calculation_errors': [],
            'duplicate_ids': [],
            'invalid_dates': [],
            'data_quality_issues': []
        }
        
        # Check for misspellings
        for correct, variants in self.bank_spellings.items():
            for variant in variants:
                if variant != correct:
                    misspelled = self.df[
                        self.df['merchant'].str.lower().str.contains(variant, na=False)
                    ]
                    if len(misspelled) > 0:
                        errors['misspellings'].extend([
                            {
                                'transaction_id': row['transaction_id'],
                                'merchant': row['merchant'],
                                'suggested_correction': correct.title()
                            }
                            for _, row in misspelled.iterrows()
                        ])
        
        # Check for calculation errors (if has_error flag or actual_amount differs)
        if 'has_error' in self.df.columns:
            calc_errors = self.df[self.df['has_error'] == True]
            errors['calculation_errors'].extend([
                {
                    'transaction_id': row['transaction_id'],
                    'displayed_amount': row['amount'],
                    'actual_amount': row.get('actual_amount', row['amount'])
                }
                for _, row in calc_errors.iterrows()
            ])
        
        # Check for duplicate transaction IDs
        duplicates = self.df[self.df.duplicated('transaction_id', keep=False)]
        if len(duplicates) > 0:
            errors['duplicate_ids'] = duplicates['transaction_id'].tolist()
        
        # Check for negative amounts (excluding refunds)
        negative_purchases = self.df[
            (self.df['amount'] < 0) &
            (~self.df['transaction_type'].str.contains('Refund', case=False, na=False))
        ]
        if len(negative_purchases) > 0:
            errors['data_quality_issues'].extend([
                {
                    'transaction_id': row['transaction_id'],
                    'issue': 'Negative amount for non-refund',
                    'amount': row['amount']
                }
                for _, row in negative_purchases.iterrows()
            ])
        
        total_errors = (
            len(errors['misspellings']) +
            len(errors['calculation_errors']) +
            len(errors['duplicate_ids']) +
            len(errors['data_quality_issues'])
        )
        
        result = {
            'requirement': 'FW40',
            'description': 'Light-touch fraud detection',
            'total_errors_found': total_errors,
            'errors_by_type': {
                'misspellings': len(errors['misspellings']),
                'calculation_errors': len(errors['calculation_errors']),
                'duplicate_ids': len(errors['duplicate_ids']),
                'data_quality_issues': len(errors['data_quality_issues'])
            },
            'details': errors,
            'fraud_risk_score': min(total_errors / 10, 10),  # 0-10 scale
            'requires_review': total_errors > 10
        }
        
        logger.info(f"FW40: Found {total_errors} potential fraud indicators")
        return result

    def analyze_fw45_gambling(self, period_months: int = 6) -> Dict[str, Any]:
        """
        FW45: Summarize gambling transactions over 6 months
        
        Analyzes:
        - Total gambling spend
        - Frequency patterns
        - Time-based analysis
        - Risk indicators
        """
        if self.df is None:
            self.load_all_data()
        
        # Detect gambling transactions
        gambling = self.df[
            self.df['merchant'].str.lower().str.contains(
                '|'.join(self.gambling_keywords), na=False
            )
        ].copy()
        
        if len(gambling) == 0:
            return {
                'requirement': 'FW45',
                'description': 'Gambling transactions (6 months)',
                'total_count': 0,
                'total_spend': 0,
                'message': 'No gambling transactions detected'
            }
        
        # Monthly breakdown
        gambling['year_month'] = gambling['date'].dt.to_period('M')
        monthly = gambling.groupby('year_month').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        monthly.columns = ['total_spend', 'transaction_count', 'avg_amount']
        
        # Detect patterns
        increasing_trend = False
        if len(monthly) >= 2:
            first_half = monthly.head(len(monthly)//2)['total_spend'].mean()
            second_half = monthly.tail(len(monthly)//2)['total_spend'].mean()
            increasing_trend = second_half > first_half * 1.2  # 20% increase
        
        result = {
            'requirement': 'FW45',
            'description': f'Gambling transactions ({period_months} months)',
            'summary': {
                'total_count': len(gambling),
                'total_spend': float(gambling['amount'].sum()),
                'avg_per_transaction': float(gambling['amount'].mean()),
                'max_single_bet': float(gambling['amount'].max()),
                'unique_operators': gambling['merchant'].nunique()
            },
            'monthly_breakdown': monthly.to_dict('index'),
            'pattern_analysis': {
                'increasing_trend': increasing_trend,
                'avg_transactions_per_month': len(gambling) / gambling['year_month'].nunique(),
                'months_with_activity': gambling['year_month'].nunique()
            },
            'transactions': gambling[
                ['transaction_id', 'date', 'amount', 'merchant']
            ].to_dict('records'),
            'risk_level': 'HIGH' if gambling['amount'].sum() > 5000 else 'MEDIUM' if gambling['amount'].sum() > 1000 else 'LOW'
        }
        
        logger.info(f"FW45: Found {len(gambling)} gambling transactions, total: £{gambling['amount'].sum():.2f}")
        return result

    def analyze_fw50_debt_payments(self, threshold: float = 500.0) -> Dict[str, Any]:
        """
        FW50: Identify and summarize large debt payments
        
        Args:
            threshold: Minimum amount to consider (default: £500)
        """
        if self.df is None:
            self.load_all_data()
        
        # Detect debt payments
        debt_mask = self.df['merchant'].str.lower().str.contains(
            '|'.join(self.debt_keywords), na=False
        )
        debt_payments = self.df[debt_mask].copy()
        
        # Filter for large payments
        large_debt = debt_payments[debt_payments['amount'] >= threshold].copy()
        
        if len(large_debt) == 0:
            return {
                'requirement': 'FW50',
                'description': f'Large debt payments (≥£{threshold})',
                'total_count': 0,
                'total_amount': 0,
                'message': 'No large debt payments detected'
            }
        
        # Group by creditor/type
        creditor_groups = large_debt.groupby('merchant').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        creditor_groups.columns = ['total_paid', 'payment_count', 'avg_payment']
        creditor_groups = creditor_groups.sort_values('total_paid', ascending=False)
        
        # Detect payment patterns
        large_debt['year_month'] = large_debt['date'].dt.to_period('M')
        monthly_totals = large_debt.groupby('year_month')['amount'].sum()
        
        result = {
            'requirement': 'FW50',
            'description': f'Large debt payments (≥£{threshold})',
            'summary': {
                'total_count': len(large_debt),
                'total_amount': float(large_debt['amount'].sum()),
                'avg_payment': float(large_debt['amount'].mean()),
                'max_payment': float(large_debt['amount'].max()),
                'unique_creditors': large_debt['merchant'].nunique()
            },
            'creditor_breakdown': creditor_groups.to_dict('index'),
            'monthly_totals': {str(k): float(v) for k, v in monthly_totals.items()},
            'transactions': large_debt[
                ['transaction_id', 'date', 'amount', 'merchant']
            ].to_dict('records'),
            'debt_burden_level': 'HIGH' if large_debt['amount'].sum() > 10000 else 'MEDIUM' if large_debt['amount'].sum() > 5000 else 'LOW'
        }
        
        logger.info(f"FW50: Found {len(large_debt)} large debt payments, total: £{large_debt['amount'].sum():.2f}")
        return result

    def analyze_all_requirements(self) -> Dict[str, Any]:
        """Run analysis for all FW requirements"""
        logger.info("Running analysis for all FW requirements...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'fw15': self.analyze_fw15_high_value(),
            'fw20': self.analyze_fw20_similar_transactions(),
            'fw25': self.analyze_fw25_missing_audit(),
            'fw30': self.analyze_fw30_missing_months(),
            'fw40': self.analyze_fw40_fraud_detection(),
            'fw45': self.analyze_fw45_gambling(),
            'fw50': self.analyze_fw50_debt_payments()
        }
        
        logger.info("Analysis complete for all FW requirements")
        return results
