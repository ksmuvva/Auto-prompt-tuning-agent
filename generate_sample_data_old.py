"""
Generate Sample Bank Transaction CSV Data
Creates 30 CSV files with realistic transaction data covering all FW requirements and edge cases

Enhanced to support:
- FW15: High-value transactions (>£250)
- FW20: Luxury brands & money transfers
- FW25: Missing audit trails
- FW30: Missing months detection
- FW40: Data errors & misspellings
- FW45: Gambling transactions
- FW50: Debt payments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import json


def generate_transaction_data(num_files=30, transactions_per_file=100):
    """Generate sample bank transaction CSV files with comprehensive edge cases"""

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # === FW20: Luxury Brands ===
    luxury_brands = [
        "Gucci London", "Louis Vuitton", "Prada", "Chanel", 
        "Rolex Boutique", "Hermes Paris", "Cartier", "Burberry",
        "Versace", "Dior", "Tiffany & Co", "Bulgari"
    ]

    # === FW20: Money Transfer Companies ===
    money_transfers = [
        "Western Union", "MoneyGram", "Wise Transfer", "PayPal Transfer",
        "Revolut Transfer", "TransferWise", "WorldRemit", "Xoom",
        "Remitly", "Azimo"
    ]

    # === FW45: Gambling Operators ===
    gambling_sites = [
        "Bet365", "William Hill", "Paddy Power", "Ladbrokes",
        "Betfair", "Sky Bet", "888 Casino", "Coral",
        "Betway", "Unibet", "Online Casino", "PokerStars"
    ]

    # === FW50: Debt/Credit Merchants ===
    debt_merchants = [
        "Barclaycard Payment", "AMEX Payment", "HSBC Loan Repayment",
        "Santander Mortgage", "Nationwide Loan", "Virgin Money Credit Card",
        "Capital One Payment", "Tesco Bank Loan", "Metro Bank Credit"
    ]

    # Regular spending
    regular_merchants = [
        # Groceries
        "Tesco Supermarket", "Sainsbury's", "Asda", "Waitrose", "Morrisons",
        "Amazon UK", "eBay", "John Lewis", "M&S", "Argos",
        
        # Transport & Fuel
        "Shell Petrol", "BP Station", "Esso", "Texaco",
        "TfL Transport", "National Rail", "Uber", "Trainline",
        
        # Food & Drink
        "Costa Coffee", "Starbucks", "Pret A Manger", "McDonald's",
        
        # Entertainment & Subscriptions
        "Netflix", "Spotify", "Sky TV", "Amazon Prime",
        
        # Utilities
        "British Gas", "Thames Water", "EDF Energy", "Octopus Energy",
        
        # Travel
        "Hotels.com", "Booking.com", "Airbnb", "British Airways", 
        "EasyJet", "Virgin Atlantic", "Premier Inn",
        
        # Electronics
        "Apple Store", "Currys PC World", "Samsung Store"
    ]

    # === FW40: Misspelled merchants (intentional errors) ===
    misspelled_merchants = [
        "Barcley Bank",  # Should be Barclays
        "HSCB Bank",     # Should be HSBC
        "Siansburys",    # Should be Sainsbury's
        "Tesoc",         # Should be Tesco
        "Amazn UK"       # Should be Amazon
    ]

    # === FW25: Merchants with missing audit trail ===
    no_audit_trail = [
        "Unknown Merchant", "Cash Withdrawal ATM", "Wire Transfer",
        "Foreign Exchange", "Anonymous Payment", "Unspecified Merchant"
    ]

    # Combine all merchants
    all_merchants = (regular_merchants + luxury_brands + money_transfers + 
                    gambling_sites + debt_merchants + misspelled_merchants + 
                    no_audit_trail)

    # Transaction types
    transaction_types = [
        "Purchase", "Online Purchase", "Contactless",
        "ATM Withdrawal", "Direct Debit", "Standing Order",
        "Bank Transfer", "Refund", "Card Payment", "Mobile Payment"
    ]

    # === FW30: Create date ranges with MISSING MONTHS (March and June) ===
    base_date = datetime(2025, 1, 1)  # Start from Jan 2025
    
    # Define allowed month ranges (excluding March and June for FW30)
    # January, February, April, May, July, August (6 months with gaps)
    allowed_months = [1, 2, 4, 5, 7, 8]
    
    print(f"Generating {num_files} CSV files with bank transaction data...")
    print(f"Coverage: 6 months with intentional gaps (March & June missing for FW30)")

    print(f"Generating {num_files} CSV files with bank transaction data...")
    print(f"Coverage: 6 months with intentional gaps (March & June missing for FW30)")

    # Ground truth tracking
    ground_truth_data = {
        "metadata": {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_files": num_files,
            "transactions_per_file": transactions_per_file,
            "missing_months": ["2025-03", "2025-06"],
            "coverage_months": ["2025-01", "2025-02", "2025-04", "2025-05", "2025-07", "2025-08"]
        },
        "fw15_high_value": [],
        "fw20_luxury_brands": [],
        "fw20_money_transfers": [],
        "fw25_missing_audit": [],
        "fw30_missing_months": ["2025-03", "2025-06"],
        "fw40_errors": [],
        "fw45_gambling": [],
        "fw50_debt_payments": []
    }

    for file_num in range(1, num_files + 1):
        transactions = []
        csv_filename = f"transactions_{file_num:02d}.csv"

        for trans_num in range(transactions_per_file):
            # Generate date in allowed months only (FW30)
            selected_month = random.choice(allowed_months)
            days_in_month = 28 if selected_month == 2 else 30
            selected_day = random.randint(1, days_in_month)
            trans_date = datetime(2025, selected_month, selected_day)

            # Determine merchant category and amount distribution
            category_roll = random.random()
            
            # === FW45: 8% Gambling ===
            if category_roll < 0.08:
                merchant = random.choice(gambling_sites)
                category = "Gambling"
                amount = round(random.uniform(10, 500), 2)
                # Note: row_number will be updated after DataFrame is sorted
                ground_truth_entry_fw45 = {
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "csv_file": csv_filename,
                    "row_number": None,  # Will be set after sorting
                    "amount": amount,
                    "merchant": merchant,
                    "category": category,
                    "date": trans_date.strftime('%Y-%m-%d'),
                    "temp_index": trans_num  # Temporary index for mapping
                }
            
            # === FW20: 5% Luxury Brands ===
            elif category_roll < 0.13:
                merchant = random.choice(luxury_brands)
                category = "Luxury"
                amount = round(random.uniform(250, 3000), 2)
                ground_truth_data["fw20_luxury_brands"].append({
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "amount": amount,
                    "merchant": merchant,
                    "date": trans_date.strftime('%Y-%m-%d')
                })
            
            # === FW20: 4% Money Transfers ===
            elif category_roll < 0.17:
                merchant = random.choice(money_transfers)
                category = "Money Transfer"
                amount = round(random.uniform(100, 2000), 2)
                ground_truth_data["fw20_money_transfers"].append({
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "amount": amount,
                    "merchant": merchant,
                    "date": trans_date.strftime('%Y-%m-%d')
                })
            
            # === FW50: 6% Debt Payments ===
            elif category_roll < 0.23:
                merchant = random.choice(debt_merchants)
                category = "Debt Payment"
                amount = round(random.uniform(100, 1500), 2)
                if amount >= 500:  # Track large debt payments
                    ground_truth_data["fw50_debt_payments"].append({
                        "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                        "amount": amount,
                        "merchant": merchant,
                        "date": trans_date.strftime('%Y-%m-%d')
                    })
            
            # === FW25: 3% Missing Audit Trail ===
            elif category_roll < 0.26:
                merchant = random.choice(no_audit_trail)
                category = "Unknown"
                amount = round(random.uniform(50, 1000), 2)
                ground_truth_data["fw25_missing_audit"].append({
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "amount": amount,
                    "merchant": merchant,
                    "date": trans_date.strftime('%Y-%m-%d'),
                    "reason": "Missing merchant details"
                })
            
            # === FW40: 2% Misspelled Merchants (Errors) ===
            elif category_roll < 0.28:
                merchant = random.choice(misspelled_merchants)
                category = "Error"
                amount = round(random.uniform(20, 200), 2)
                ground_truth_data["fw40_errors"].append({
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "error_type": "misspelling",
                    "merchant": merchant,
                    "date": trans_date.strftime('%Y-%m-%d')
                })
            
            # === Regular transactions (72%) ===
            else:
                merchant = random.choice(regular_merchants)
                category = categorize_merchant(merchant)
                
                # Amount distribution for regular transactions
                if random.random() < 0.15:  # 15% high-value
                    amount = round(random.uniform(250, 1000), 2)
                elif random.random() < 0.05:  # 5% very high (anomalies)
                    amount = round(random.uniform(1000, 5000), 2)
                else:  # 80% normal
                    amount = round(random.uniform(5, 249), 2)

            # Occasionally negative (refunds)
            if random.random() < 0.05:
                amount = -abs(amount)

            # === FW15: Track high-value transactions ===
            if amount > 250:
                ground_truth_data["fw15_high_value"].append({
                    "transaction_id": f"TXN{file_num:03d}{trans_num:04d}",
                    "amount": amount,
                    "merchant": merchant,
                    "category": category,
                    "date": trans_date.strftime('%Y-%m-%d')
                })

            trans_type = random.choice(transaction_types)

            # Generate transaction ID
            trans_id = f"TXN{file_num:03d}{trans_num:04d}"

            # === FW40: Add calculation errors randomly (1% of transactions) ===
            display_amount = amount
            has_error = False
            if random.random() < 0.01:
                # Introduce decimal errors
                display_amount = amount * 10 if amount < 100 else amount / 10
                has_error = True
                ground_truth_data["fw40_errors"].append({
                    "transaction_id": trans_id,
                    "error_type": "calculation_error",
                    "correct_amount": amount,
                    "displayed_amount": display_amount,
                    "date": trans_date.strftime('%Y-%m-%d')
                })

            # Random description
            descriptions = [
                f"{merchant}",
                f"{merchant} - {trans_type}",
                f"Payment to {merchant}",
                f"{trans_type} - {merchant}"
            ]

            transaction = {
                'transaction_id': trans_id,
                'date': trans_date.strftime('%Y-%m-%d'),
                'amount': display_amount,  # May have errors (FW40)
                'actual_amount': amount,  # True amount for validation
                'currency': 'GBP',
                'merchant': merchant,
                'transaction_type': trans_type,
                'description': random.choice(descriptions),
                'category': category,
                'status': 'Completed' if random.random() > 0.02 else random.choice(['Pending', 'Failed']),
                'has_error': has_error
            }

            # Add contextual notes
            if amount > 1000:
                transaction['notes'] = 'High value transaction'
            elif merchant in gambling_sites:
                transaction['notes'] = 'Gambling transaction'
            elif merchant in no_audit_trail:
                transaction['notes'] = 'Missing audit trail'
            elif merchant in luxury_brands:
                transaction['notes'] = 'Luxury purchase'
            elif has_error:
                transaction['notes'] = 'Data error detected'
            else:
                transaction['notes'] = ''

            transactions.append(transaction)

        # Create DataFrame
        df = pd.DataFrame(transactions)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Save to CSV
        filename = data_dir / f"transactions_{file_num:02d}.csv"
        df.to_csv(filename, index=False)

        print(f"  Created {filename.name} ({len(df)} transactions)")

    # === Save Ground Truth Master File ===
    ground_truth_file = data_dir / "ground_truth_master.json"
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    print(f"\n✓ Generated {num_files} CSV files in '{data_dir}' directory")
    print(f"✓ Saved ground truth to {ground_truth_file}")
    print(f"  Total transactions: {num_files * transactions_per_file}")

    # Generate summary statistics
    all_files = list(data_dir.glob("transactions_*.csv"))
    all_data = pd.concat([pd.read_csv(f) for f in all_files])

    print("\n" + "="*60)
    print("DATASET SUMMARY - FW REQUIREMENTS COVERAGE")
    print("="*60)
    print(f"  Total records: {len(all_data)}")
    print(f"  Date range: {all_data['date'].min()} to {all_data['date'].max()}")
    print(f"  Amount range: £{all_data['amount'].min():.2f} to £{all_data['amount'].max():.2f}")
    print(f"\nFW15 - High-Value Transactions (>£250):")
    print(f"  Count: {len(ground_truth_data['fw15_high_value'])}")
    print(f"\nFW20 - Luxury Brands:")
    print(f"  Count: {len(ground_truth_data['fw20_luxury_brands'])}")
    print(f"  Total Amount: £{sum(t['amount'] for t in ground_truth_data['fw20_luxury_brands']):.2f}")
    print(f"\nFW20 - Money Transfers:")
    print(f"  Count: {len(ground_truth_data['fw20_money_transfers'])}")
    print(f"\nFW25 - Missing Audit Trail:")
    print(f"  Count: {len(ground_truth_data['fw25_missing_audit'])}")
    print(f"\nFW30 - Missing Months:")
    print(f"  Missing: {', '.join(ground_truth_data['fw30_missing_months'])}")
    print(f"  Coverage: {', '.join(ground_truth_data['metadata']['coverage_months'])}")
    print(f"\nFW40 - Data Errors:")
    print(f"  Count: {len(ground_truth_data['fw40_errors'])}")
    print(f"  Types: Misspellings, Calculation errors")
    print(f"\nFW45 - Gambling Transactions:")
    print(f"  Count: {len(ground_truth_data['fw45_gambling'])}")
    print(f"  Total Amount: £{sum(t['amount'] for t in ground_truth_data['fw45_gambling']):.2f}")
    print(f"\nFW50 - Large Debt Payments (≥£500):")
    print(f"  Count: {len(ground_truth_data['fw50_debt_payments'])}")
    print(f"  Total Amount: £{sum(t['amount'] for t in ground_truth_data['fw50_debt_payments']):.2f}")
    print(f"\n  Unique merchants: {all_data['merchant'].nunique()}")
    print("="*60)


def categorize_merchant(merchant):
    """Categorize merchant for better organization"""
    categories = {
        'Groceries': ['Tesco', 'Sainsbury', 'Asda', 'Waitrose', 'M&S', 'Morrisons'],
        'Online Shopping': ['Amazon', 'eBay', 'John Lewis'],
        'Transport': ['Shell', 'BP', 'Esso', 'Texaco', 'TfL', 'National Rail', 'Uber', 'Trainline'],
        'Food & Drink': ['Costa', 'Starbucks', 'Pret', 'McDonald'],
        'Entertainment': ['Netflix', 'Spotify', 'Sky TV'],
        'Gambling': ['Casino', 'Bet365', 'William Hill', 'Paddy Power', 'Ladbrokes', 
                     'Betfair', 'Sky Bet', '888', 'Coral', 'Betway', 'Unibet', 'PokerStars'],
        'Utilities': ['British Gas', 'Thames Water', 'EDF Energy', 'Octopus Energy'],
        'Travel': ['Hotels', 'Booking', 'Airbnb', 'Airways', 'EasyJet', 'Virgin', 'Premier Inn'],
        'Electronics': ['Apple', 'Currys', 'Argos', 'Samsung'],
        'Luxury': ['Gucci', 'Louis Vuitton', 'Prada', 'Chanel', 'Rolex', 'Hermes', 
                   'Cartier', 'Burberry', 'Versace', 'Dior', 'Tiffany', 'Bulgari'],
        'Money Transfer': ['Western Union', 'MoneyGram', 'Wise', 'PayPal Transfer', 
                          'Revolut', 'TransferWise', 'WorldRemit', 'Xoom', 'Remitly', 'Azimo'],
        'Debt Payment': ['Barclaycard', 'AMEX', 'HSBC Loan', 'Santander Mortgage', 
                         'Nationwide Loan', 'Virgin Money', 'Capital One', 'Tesco Bank', 'Metro Bank'],
        'Unknown': ['Unknown', 'Cash Withdrawal', 'Wire Transfer', 'Foreign Exchange', 'Anonymous'],
        'Error': ['Barcley', 'HSCB', 'Siansburys', 'Tesoc', 'Amazn'],
        'Other': []
    }

    for category, keywords in categories.items():
        if any(keyword in merchant for keyword in keywords):
            return category

    return 'Other'


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED DATA GENERATOR - FW REQUIREMENTS")
    print("="*60)
    generate_transaction_data(num_files=30, transactions_per_file=100)
    print("\n✓ Sample data generation complete!")
    print("✓ Ground truth master file created")
    print("✓ All FW requirements covered (FW15-FW50)")
    print("="*60 + "\n")
