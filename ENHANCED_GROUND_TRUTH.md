# Enhanced Ground Truth System

**Version 2.0 - TRUE Mathematical Metrics Support**

## Overview

The enhanced ground truth system provides comprehensive tracking for all 7 FW requirements with detailed information that enables human verification and TRUE mathematical metrics calculation (TP/TN/FP/FN).

## Key Enhancements

### 1. **CSV File Names**
Every detection now includes the exact CSV file where the transaction appears.

```json
{
  "transaction_id": "TXN0010017",
  "csv_file": "transactions_01.csv",
  ...
}
```

### 2. **Row Numbers**
Exact row numbers (accounting for header row) for easy human verification.

```json
{
  "transaction_id": "TXN0010017",
  "csv_file": "transactions_01.csv",
  "row_number": 5,
  ...
}
```

### 3. **Full Transaction Data**
Complete transaction details for cross-checking LLM outputs.

```json
{
  "full_data": {
    "transaction_id": "TXN0010017",
    "date": "2025-01-11",
    "amount": 448.07,
    "merchant": "Pret A Manger",
    "category": "Food & Drink",
    "description": "Payment to Pret A Manger",
    "transaction_type": "Contactless",
    "status": "Completed"
  }
}
```

### 4. **Verification Notes**
Clear instructions for human verification.

```json
{
  "verification_note": "Check transactions_01.csv row 5"
}
```

## FW Requirements Coverage

### FW15: High-Value Transactions (>£250)
**Count:** 989 transactions
**Enhanced Fields:**
- `csv_file`: CSV file name
- `row_number`: Exact row in CSV
- `amount`: Transaction amount
- `merchant`: Merchant name
- `category`: Transaction category
- `date`: Transaction date
- `full_data`: Complete transaction details
- `verification_note`: Human verification instruction

**Example:**
```json
{
  "transaction_id": "TXN0010017",
  "csv_file": "transactions_01.csv",
  "row_number": 5,
  "amount": 448.07,
  "merchant": "Pret A Manger",
  "category": "Food & Drink",
  "date": "2025-01-11",
  "full_data": { ... },
  "verification_note": "Check transactions_01.csv row 5"
}
```

### FW20: Luxury Brands
**Count:** 138 transactions
**Total Amount:** £212,069.25
**Enhanced Fields:** Same as FW15

### FW20: Money Transfers
**Count:** 136 transactions
**Enhanced Fields:** Same as FW15

### FW25: Missing Audit Trail
**Count:** 83 transactions
**Enhanced Fields:**
- All FW15 fields
- `reason`: "Missing merchant details"

### FW30: Missing Months
**Missing:** March (2025-03), June (2025-06)
**Coverage:** January, February, April, May, July, August

### FW40: Data Errors
**Count:** 96 errors
**Types:**
1. **Misspellings** - Merchant names with intentional typos
2. **Calculation Errors** - Amount discrepancies

**Enhanced Fields:**
- `error_type`: "misspelling" or "calculation_error"
- `merchant`: Misspelled merchant name (for misspellings)
- `correct_amount`: True amount (for calculations)
- `displayed_amount`: Displayed (incorrect) amount (for calculations)
- `verification_note`: Specific error description

**Example (Misspelling):**
```json
{
  "transaction_id": "TXN0120045",
  "csv_file": "transactions_12.csv",
  "row_number": 23,
  "error_type": "misspelling",
  "merchant": "Tesoc",
  "date": "2025-02-15",
  "full_data": { ... },
  "verification_note": "Check transactions_12.csv row 23 - merchant name misspelled"
}
```

**Example (Calculation Error):**
```json
{
  "transaction_id": "TXN0150078",
  "csv_file": "transactions_15.csv",
  "row_number": 41,
  "error_type": "calculation_error",
  "correct_amount": 45.50,
  "displayed_amount": 455.00,
  "date": "2025-04-20",
  "full_data": { ... },
  "verification_note": "Check transactions_15.csv row 41 - amount calculation error"
}
```

### FW45: Gambling Transactions
**Count:** 272 transactions
**Total Amount:** £61,816.50
**Enhanced Fields:** Same as FW15

### FW50: Large Debt Payments (≥£500)
**Count:** 134 transactions
**Total Amount:** £116,279.95
**Enhanced Fields:** Same as FW15

## Human Verification Process

### Step-by-Step Verification

1. **Read Ground Truth Entry**
```json
{
  "transaction_id": "TXN0010017",
  "csv_file": "transactions_01.csv",
  "row_number": 5,
  "amount": 448.07,
  "merchant": "Pret A Manger"
}
```

2. **Open CSV File**
```bash
cat data/transactions_01.csv | head -6 | tail -2
```

3. **Verify Row 5**
```
TXN0010017,2025-01-11,448.07,448.07,GBP,Pret A Manger,Contactless,Payment to Pret A Manger,Food & Drink,Completed,False,FW15,
```

4. **Confirm Match**
- ✅ Transaction ID matches: TXN0010017
- ✅ Amount matches: 448.07
- ✅ Merchant matches: Pret A Manger
- ✅ Date matches: 2025-01-11
- ✅ Marked as FW15 (high-value >£250)

## TRUE Mathematical Metrics

### Confusion Matrix Calculation

With the enhanced ground truth, you can calculate exact TP/TN/FP/FN:

**Definitions:**
- **TP (True Positives)**: LLM correctly identified transactions in ground truth
- **FP (False Positives)**: LLM identified transactions NOT in ground truth
- **FN (False Negatives)**: LLM missed transactions that ARE in ground truth
- **TN (True Negatives)**: LLM correctly did NOT identify transactions not in ground truth

**Example Calculation:**

```
Total Transactions: 3000
Ground Truth High-Value (FW15): 989

LLM Detected: 1005 transactions
```

**Step 1: Extract LLM Response Transaction IDs**
```python
llm_ids = extract_transaction_ids_from_llm_response(llm_response)
# Returns: {'TXN0010017', 'TXN0010008', ...}  (1005 IDs)
```

**Step 2: Load Ground Truth IDs**
```python
ground_truth_ids = {t['transaction_id'] for t in ground_truth['fw15_high_value']}
# Returns: {'TXN0010017', 'TXN0010008', ...}  (989 IDs)
```

**Step 3: Calculate Confusion Matrix**
```python
TP = len(llm_ids.intersection(ground_truth_ids))  # 970
FP = len(llm_ids - ground_truth_ids)               # 35
FN = len(ground_truth_ids - llm_ids)               # 19
TN = 3000 - TP - FP - FN                           # 1976
```

**Step 4: Calculate Metrics**
```python
Precision = TP / (TP + FP) = 970 / 1005 = 96.52%
Recall    = TP / (TP + FN) = 970 / 989  = 98.08%
Accuracy  = (TP + TN) / Total = (970 + 1976) / 3000 = 98.20%
F1 Score  = 2 * (P * R) / (P + R) = 97.29%
```

### Human Verification of False Positives

If LLM reports a transaction that's NOT in ground truth:

```python
fp_id = "TXN0050023"  # LLM said this is high-value, but it's not in ground truth
```

**Verify manually:**
1. Find transaction in ground truth: `grep TXN0050023 data/ground_truth_master.json`
2. Check CSV file directly: `grep TXN0050023 data/transactions_05.csv`
3. Confirm amount is actually ≤£250 (not high-value)

**Result:** LLM incorrectly identified this as high-value → True False Positive

## Code Examples

### Loading Enhanced Ground Truth

```python
import json

with open('data/ground_truth_master.json') as f:
    ground_truth = json.load(f)

# Access FW15 detections
for detection in ground_truth['fw15_high_value']:
    print(f"Transaction: {detection['transaction_id']}")
    print(f"Location: {detection['csv_file']} row {detection['row_number']}")
    print(f"Amount: £{detection['amount']:.2f}")
    print(f"Verify: {detection['verification_note']}")
    print()
```

### Verifying Against CSV

```python
import pandas as pd

# Load ground truth
with open('data/ground_truth_master.json') as f:
    gt = json.load(f)

# Get first FW15 detection
detection = gt['fw15_high_value'][0]

# Load the CSV file
df = pd.read_csv(f"data/{detection['csv_file']}")

# Get the specific row (row_number is 1-indexed, DataFrame is 0-indexed, +1 for header)
row = df.iloc[detection['row_number'] - 2]

# Verify
assert row['transaction_id'] == detection['transaction_id']
assert float(row['amount']) == detection['amount']
assert row['merchant'] == detection['merchant']

print(f"✅ Verified: {detection['transaction_id']} in {detection['csv_file']} row {detection['row_number']}")
```

### TRUE Metrics Calculation

```python
from agent.true_metrics import TrueMetricsCalculator

calculator = TrueMetricsCalculator()

# LLM response
llm_response = """
Found 1005 high-value transactions:
- TXN0010017: £448.07
- TXN0010008: £1,258.43
...
"""

# Load ground truth
with open('data/ground_truth_master.json') as f:
    ground_truth = json.load(f)

ground_truth_ids = {t['transaction_id'] for t in ground_truth['fw15_high_value']}

# Calculate metrics
metrics = calculator.calculate_metrics(
    llm_response=llm_response,
    ground_truth={'high_value_transactions': list(ground_truth_ids)},
    total_transactions=3000
)

print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2%}")
```

## Benefits

### For Development
1. **Debugging**: Quickly find specific transactions causing false positives/negatives
2. **Testing**: Verify each detection manually if needed
3. **Validation**: Ensure ground truth is accurate

### For Users
1. **Trust**: Can verify any detection manually
2. **Transparency**: See exactly where each detection comes from
3. **Compliance**: Audit trail for regulatory requirements

### For TRUE AI Agent
1. **Adaptive Learning**: Precise feedback on what worked/didn't work
2. **TRUE Metrics**: Exact TP/TN/FP/FN calculations
3. **Iterative Improvement**: Know exactly which transactions to focus on

## Metadata

```json
{
  "metadata": {
    "version": "2.0",
    "created": "2025-10-23T...",
    "total_files": 30,
    "transactions_per_file": 100,
    "missing_months": ["2025-03", "2025-06"],
    "coverage_months": ["2025-01", "2025-02", "2025-04", "2025-05", "2025-07", "2025-08"],
    "enhancements": [
      "CSV file names included",
      "Row numbers for human verification",
      "Full transaction data for cross-checking",
      "TRUE mathematical metrics support"
    ]
  }
}
```

## Generating New Data

To regenerate the enhanced dataset:

```bash
python generate_sample_data.py
```

This will:
1. Create 30 CSV files with 100 transactions each (3,000 total)
2. Generate enhanced ground truth with CSV files and row numbers
3. Cover all 7 FW requirements comprehensively
4. Create verification notes for each detection

## Summary

The enhanced ground truth system provides:

✅ **CSV file names** - Know exactly where each transaction is
✅ **Row numbers** - Verify any detection in seconds
✅ **Full transaction data** - Complete context for each detection
✅ **Verification notes** - Clear instructions for manual checks
✅ **TRUE metrics support** - Exact TP/TN/FP/FN calculations
✅ **Human verifiable** - Anyone can audit the results
✅ **All FW requirements** - Comprehensive coverage (FW15-FW50)

This enables both the Legacy System and TRUE AI Agent to use **real mathematical metrics** instead of heuristics, providing precise measurement of detection accuracy.

---

**Generated:** 2025-10-23
**Version:** 2.0
**Supports:** Legacy System & TRUE AI Agent
