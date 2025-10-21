"""
Generate Sample Bank Transaction CSV Data
Creates 30 CSV files with realistic transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


def generate_transaction_data(num_files=30, transactions_per_file=100):
    """Generate sample bank transaction CSV files"""

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Merchant categories
    merchants = [
        # Regular spending
        "Tesco Supermarket", "Sainsbury's", "Asda", "Waitrose",
        "Amazon UK", "eBay", "John Lewis", "M&S",
        "Shell Petrol", "BP Station", "Esso",
        "Costa Coffee", "Starbucks", "Pret A Manger",
        "TfL Transport", "National Rail", "Uber",
        "Netflix", "Spotify", "Sky TV",
        "British Gas", "Thames Water", "EDF Energy",

        # High-value merchants
        "Apple Store", "Currys PC World", "Argos",
        "Hotels.com", "Booking.com", "Airbnb",
        "British Airways", "EasyJet", "Virgin Atlantic",

        # Potentially suspicious
        "Foreign Exchange", "Wire Transfer", "Cash Withdrawal",
        "Online Casino", "Bet365", "Unknown Merchant"
    ]

    # Transaction types
    transaction_types = [
        "Purchase", "Online Purchase", "Contactless",
        "ATM Withdrawal", "Direct Debit", "Standing Order",
        "Bank Transfer", "Refund"
    ]

    start_date = datetime.now() - timedelta(days=365)

    print(f"Generating {num_files} CSV files with bank transaction data...")

    for file_num in range(1, num_files + 1):
        transactions = []

        for trans_num in range(transactions_per_file):
            # Random date within last year
            days_ago = random.randint(0, 365)
            trans_date = start_date + timedelta(days=days_ago)

            # Transaction amount - most are normal, some are high-value
            if random.random() < 0.15:  # 15% high-value transactions
                amount = round(random.uniform(250, 2000), 2)
            elif random.random() < 0.05:  # 5% very high (anomalies)
                amount = round(random.uniform(2000, 10000), 2)
            else:  # 80% normal transactions
                amount = round(random.uniform(5, 249), 2)

            # Occasionally negative (refunds)
            if random.random() < 0.05:
                amount = -abs(amount)

            merchant = random.choice(merchants)
            trans_type = random.choice(transaction_types)

            # Generate transaction ID
            trans_id = f"TXN{file_num:03d}{trans_num:04d}"

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
                'amount': amount,
                'currency': 'GBP',
                'merchant': merchant,
                'transaction_type': trans_type,
                'description': random.choice(descriptions),
                'category': categorize_merchant(merchant),
                'status': 'Completed' if random.random() > 0.02 else random.choice(['Pending', 'Failed'])
            }

            # Add some anomaly indicators for testing
            if amount > 1000:
                transaction['notes'] = 'High value transaction'
            elif merchant in ['Online Casino', 'Bet365', 'Unknown Merchant']:
                transaction['notes'] = 'Flagged for review'
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

        print(f"  Created {filename} ({len(df)} transactions)")

    print(f"\n✓ Generated {num_files} CSV files in '{data_dir}' directory")
    print(f"  Total transactions: {num_files * transactions_per_file}")

    # Generate summary statistics
    all_files = list(data_dir.glob("transactions_*.csv"))
    all_data = pd.concat([pd.read_csv(f) for f in all_files])

    print("\nDATASET SUMMARY:")
    print(f"  Total records: {len(all_data)}")
    print(f"  Date range: {all_data['date'].min()} to {all_data['date'].max()}")
    print(f"  Amount range: £{all_data['amount'].min():.2f} to £{all_data['amount'].max():.2f}")
    print(f"  Transactions > 250 GBP: {len(all_data[all_data['amount'] > 250])}")
    print(f"  Transactions > 1000 GBP: {len(all_data[all_data['amount'] > 1000])}")
    print(f"  Unique merchants: {all_data['merchant'].nunique()}")


def categorize_merchant(merchant):
    """Categorize merchant for better organization"""
    categories = {
        'Groceries': ['Tesco', 'Sainsbury', 'Asda', 'Waitrose', 'M&S'],
        'Online Shopping': ['Amazon', 'eBay', 'John Lewis'],
        'Transport': ['Shell', 'BP', 'Esso', 'TfL', 'National Rail', 'Uber'],
        'Food & Drink': ['Costa', 'Starbucks', 'Pret'],
        'Entertainment': ['Netflix', 'Spotify', 'Sky TV', 'Casino', 'Bet365'],
        'Utilities': ['British Gas', 'Thames Water', 'EDF Energy'],
        'Travel': ['Hotels', 'Booking', 'Airbnb', 'Airways', 'EasyJet', 'Virgin'],
        'Electronics': ['Apple', 'Currys', 'Argos'],
        'Other': []
    }

    for category, keywords in categories.items():
        if any(keyword in merchant for keyword in keywords):
            return category

    return 'Other'


if __name__ == "__main__":
    generate_transaction_data(num_files=30, transactions_per_file=100)
    print("\n✓ Sample data generation complete!")
