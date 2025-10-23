# Data Generation Quality Issues - Detailed Analysis

**Comprehensive analysis of data generation quality problems**

Date: 2025-10-23

---

## 🎯 EXECUTIVE SUMMARY

The synthetic data generator successfully generates data, but **data quality is only 50-60%** of production requirements.

**Major Issues:**
1. Type inconsistencies (integers as floats, strings as numbers)
2. Unrealistic values (mock data artifacts)
3. No cross-field validation
4. No domain knowledge
5. No statistical realism

---

## 🔴 CRITICAL DATA QUALITY ISSUES

### 1. **Type Inconsistencies**

#### Issue: Wrong Data Types Generated

**Examples from Test Output:**
```python
# Age should be integer, got float:
{'age': 38.38}

# Customer ID should be string, got integer:
{'customer_id': 86}

# Name should be string, got integer:
{'name': 73}
```

**Impact:** Data is unusable without post-processing

**Root Cause:**
```python
# In reasoning_engines.py - Monte Carlo engine
value = np.random.normal(mean, std)
sample[field] = round(value, 2) if field_type == 'number' else int(value)
#                      ^^^^ WRONG! Should check semantic type (age, count)
```

**Should Be:**
```python
# Semantic type checking
if field in ['age', 'count', 'quantity', 'years']:
    sample[field] = int(value)  # Always integer
elif field in ['price', 'amount', 'percentage']:
    sample[field] = round(value, 2)  # 2 decimal places
elif field in ['weight', 'measurement']:
    sample[field] = round(value, 3)  # 3 decimal places
```

---

### 2. **Unrealistic Values**

#### Issue: Mock Provider Generates Nonsensical Data

**Examples:**
```python
# Mock provider output:
{
    'customer_id': 'mock_customer_id',  # Not realistic
    'name': 'mock_name',                 # Unusable
    'email': 'mock_email',               # Invalid email
    'age': 42                            # Default for everything
}
```

**Impact:** Tests pass but data is worthless

**Root Cause:** Mock provider has no domain knowledge

**Solution Needed:**
```python
class MockProvider:
    def __init__(self):
        self.realistic_defaults = {
            'customer': {
                'id': lambda: f'CUST{random.randint(1000, 9999)}',
                'name': lambda: random.choice(['John Smith', 'Jane Doe']),
                'email': lambda: f'user{random.randint(1, 999)}@example.com',
                'age': lambda: random.randint(18, 80)
            }
        }
```

---

### 3. **No Cross-Field Validation**

#### Issue: Generated Fields Don't Make Sense Together

**Examples of Bad Data:**
```python
{
    'birth_date': '2010-05-15',
    'age': 55,                    # Contradictory!
    'employment_start': '2005-01-01'  # Started work at age 0!
}

{
    'country': 'UK',
    'state': 'California',        # California not in UK!
    'postcode': 'SW1A 1AA'
}

{
    'salary': 25000,
    'job_title': 'CEO',           # CEO salary too low!
    'years_experience': 20
}
```

**Impact:** Data fails real-world validation checks

**What's Missing:**
```python
# MISSING: Consistency rules
class ConsistencyValidator:
    def validate_age_vs_birth_date(self, data):
        age = calculate_age(data['birth_date'])
        assert abs(age - data['age']) <= 1, "Age doesn't match birth date"

    def validate_geography(self, data):
        country = data['country']
        state = data.get('state')
        postcode = data.get('postcode')

        if country == 'UK':
            assert state is None, "UK doesn't have states"
            assert is_valid_uk_postcode(postcode)
```

---

### 4. **No Domain Knowledge**

#### Issue: Generated Data Lacks Domain-Specific Realism

**Healthcare Example:**
```python
# What we generate:
{
    'diagnosis_code': 'mock_diagnosis_code',  # Not ICD-10
    'medication': 'mock_medication',          # Not real drug name
    'lab_test': 'mock_lab_test',             # Not LOINC code
    'result': 42                              # No units, no ref range
}

# What we should generate:
{
    'diagnosis_code': 'E11.9',               # Type 2 diabetes
    'diagnosis_name': 'Type 2 diabetes mellitus without complications',
    'medication': 'Metformin 500mg',
    'lab_test': 'HbA1c',
    'result': 7.2,
    'units': '%',
    'reference_range': '4.0-6.0',
    'interpretation': 'High'
}
```

**Impact:** Data rejected by domain experts

**What's Missing:**
```python
# MISSING: Domain knowledge databases
DOMAIN_KNOWLEDGE = {
    'healthcare': {
        'icd10_codes': {
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential (primary) hypertension',
            # ... 70,000+ codes
        },
        'medications': [
            {'name': 'Metformin', 'dosages': ['500mg', '850mg', '1000mg']},
            # ... thousands more
        ],
        'lab_tests': {
            'HbA1c': {'units': '%', 'ref_range': (4.0, 6.0)},
            'Glucose': {'units': 'mmol/L', 'ref_range': (4.0, 5.9)}
        }
    },
    'finance': {
        'transaction_types': ['PURCHASE', 'REFUND', 'TRANSFER', 'PAYMENT'],
        'currency_codes': ['GBP', 'USD', 'EUR', 'JPY'],
        # ...
    }
}
```

---

### 5. **No Statistical Realism**

#### Issue: Distributions Don't Match Real World

**Problem 1: Uniform When Should Be Skewed**
```python
# What we generate (uniform age distribution):
ages = [18, 25, 32, 45, 51, 68, 72, 85]  # Equal probability

# Real world (normal distribution):
ages = [25, 28, 30, 31, 32, 33, 35, 38]  # Clustered around mean
```

**Problem 2: No Correlations**
```python
# What we generate (independent):
{
    'age': 25,
    'salary': 150000,  # Unlikely for age 25!
    'job_level': 'Senior Executive'
}

# Should be correlated:
{
    'age': 25,
    'salary': 35000,   # More realistic
    'job_level': 'Junior'
}
```

**Problem 3: No Temporal Patterns**
```python
# What we generate (random dates):
order_dates = ['2025-01-05', '2025-08-20', '2025-03-12', '2025-11-30']

# Real world (seasonal patterns):
# E-commerce: More orders in November-December (holidays)
# B2B: Dips in August (summer holidays), December (year-end)
```

**What's Missing:**
```python
# MISSING: Correlation models
class CorrelationEngine:
    def define_correlation(self, field1, field2, strength, type):
        """
        Define how two fields correlate

        Examples:
        - age vs salary: positive correlation (0.6)
        - temperature vs heating_cost: negative correlation (-0.8)
        """
        pass

    def apply_correlations(self, data):
        """Adjust data to respect correlations"""
        pass

# MISSING: Temporal patterns
class TemporalPatternEngine:
    def add_seasonal_pattern(self, field, pattern):
        """
        pattern = {
            'winter': 1.5,   # 50% increase
            'summer': 0.7,   # 30% decrease
            'spring': 1.0,
            'autumn': 1.2
        }
        """
        pass
```

---

### 6. **No Relationship Modeling**

#### Issue: Cannot Generate Related Records

**Example: Customer + Orders**
```python
# What we generate now (disconnected):
customers = generate_data('customer', count=100)
orders = generate_data('order', count=500)

# Problem: Orders don't reference customers!
# All customer_id fields are random
```

**What We Need:**
```python
# MISSING: Relationship modeling
generator = SyntheticDataGenerator()

# Define schema with relationships
generator.define_entity('customer', {
    'id': 'string',
    'name': 'string',
    'email': 'email'
})

generator.define_entity('order', {
    'id': 'string',
    'customer_id': 'foreign_key',  # References customer
    'amount': 'currency',
    'date': 'date'
})

generator.define_relationship(
    parent='customer',
    child='order',
    type='one_to_many',
    foreign_key='customer_id',
    cardinality=(1, 10)  # Each customer has 1-10 orders
)

# Generate related data
data = generator.generate_related_data({
    'customer': 100,
    'order': 500  # Distributed across 100 customers
})
```

---

### 7. **No Hierarchical Data Support**

#### Issue: Cannot Generate Tree Structures

**Example: Organization → Department → Employee**
```python
# MISSING: Hierarchical generation
organization = {
    'id': 'ORG001',
    'name': 'Acme Corp',
    'departments': [
        {
            'id': 'DEPT001',
            'name': 'Engineering',
            'employees': [
                {'id': 'EMP001', 'name': 'John', 'role': 'Engineer'},
                {'id': 'EMP002', 'name': 'Jane', 'role': 'Manager'}
            ]
        },
        {
            'id': 'DEPT002',
            'name': 'Sales',
            'employees': [...]
        }
    ]
}
```

---

### 8. **Poor Diversity**

#### Issue: Generated Data Too Similar

**Examples:**
```python
# Email pattern repeats:
emails = [
    'user1@example.com',
    'user2@example.com',
    'user3@example.com',
    # All same pattern!
]

# Names not diverse:
names = [
    'John Smith',
    'Jane Smith',
    'Bob Smith',
    # All same surname!
]

# Geographic concentration:
postcodes = [
    'SW1A 1AA',
    'SW1A 2BB',
    'SW1A 3CC',
    # All Westminster, London!
]
```

**What's Missing:**
```python
# MISSING: Diversity metrics and enforcement
class DiversityEngine:
    def calculate_diversity(self, data, field):
        """
        Calculate Shannon entropy or Gini coefficient
        """
        pass

    def ensure_minimum_diversity(self, data, field, min_threshold):
        """
        Ensure at least X% of values are unique
        """
        pass

    def balance_distribution(self, data, field, target_distribution):
        """
        Ensure distribution matches target
        Example: 80% British, 9% Asian, 4% Black, 3% Mixed, 4% Other
        """
        pass
```

---

### 9. **No Anomaly Injection**

#### Issue: All Data Is "Normal" - No Edge Cases

**Real datasets have:**
- Outliers (very old customers, very high spenders)
- Edge cases (empty strings, minimum values)
- Errors (typos, missing data)
- Duplicates (intentional or not)

**What's Missing:**
```python
# MISSING: Anomaly injection
class AnomalyInjector:
    def inject_outliers(self, data, field, percentage=0.05):
        """Add 5% outliers to a numeric field"""
        pass

    def inject_missing_data(self, data, field, percentage=0.02):
        """Make 2% of field values null/empty"""
        pass

    def inject_typos(self, data, field, percentage=0.01):
        """Add 1% typos (Tesoc instead of Tesco)"""
        pass

    def inject_duplicates(self, data, percentage=0.03):
        """Make 3% of records duplicates"""
        pass
```

---

### 10. **Format Validation Issues**

#### Issue: Generated Data Fails Format Validation

**Examples:**
```python
# Invalid email formats generated:
'john smith@example.com'  # Space not allowed
'test@'                   # Missing domain
'@example.com'            # Missing local part

# Invalid UK postcodes:
'SW1A1AA'                 # Missing space
'SW1A 1A'                 # Incomplete inward code
'XX1A 1AA'                # Invalid area code

# Invalid phone numbers:
'07700-900-123'           # Wrong separators
'+44 7700900123'          # No spacing
'00447700900123'          # Wrong prefix
```

**Root Cause:** No format validation after generation

**What's Missing:**
```python
# MISSING: Format validators
class FormatValidator:
    @staticmethod
    def validate_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_uk_postcode(postcode):
        # Proper UK postcode validation
        pass

    @staticmethod
    def fix_or_regenerate(value, validator):
        """Try to fix format, or regenerate if can't fix"""
        pass
```

---

## 📊 DATA QUALITY METRICS

### Current Quality Scores (Estimated):

| Aspect | Score | Target | Gap |
|--------|-------|--------|-----|
| Type Correctness | 60% | 100% | -40% |
| Value Realism | 40% | 95% | -55% |
| Format Validity | 70% | 100% | -30% |
| Cross-Field Consistency | 0% | 95% | -95% |
| Domain Accuracy | 0% | 90% | -90% |
| Statistical Realism | 30% | 85% | -55% |
| Diversity | 50% | 80% | -30% |
| Relationship Accuracy | 0% | 90% | -90% |
| **Overall Quality** | **35%** | **90%** | **-55%** |

---

## 🔧 SPECIFIC EXAMPLES OF BAD DATA

### Example 1: Customer Data Generated

**What We Generate:**
```json
{
  "customer_id": 86,
  "name": 73,
  "email": 42,
  "age": 38.38,
  "postcode": "SW1A 1AA",
  "phone": "07700 900 123"
}
```

**What We SHOULD Generate:**
```json
{
  "customer_id": "CUST0086",
  "name": "Emma Thompson",
  "email": "emma.thompson@gmail.com",
  "age": 38,
  "postcode": "SW1A 1AA",
  "phone": "07700 900 123"
}
```

**Issues:**
- ✗ ID is integer not string
- ✗ Name is integer not string
- ✗ Email is integer not email
- ✗ Age has decimal places

---

### Example 2: Healthcare Data Generated

**What We Generate:**
```json
{
  "patient_id": "mock_patient_id",
  "diagnosis": "mock_diagnosis",
  "medication": "mock_medication",
  "test_result": 42
}
```

**What We SHOULD Generate:**
```json
{
  "patient_id": "NHS-GB-9434765919",
  "diagnosis_code": "E11.9",
  "diagnosis_name": "Type 2 diabetes mellitus without complications",
  "medication": "Metformin 500mg twice daily",
  "medication_code": "BNF-0601022B0",
  "test_name": "HbA1c",
  "test_code": "LOINC-4548-4",
  "test_result": 7.2,
  "test_units": "%",
  "reference_range": "4.0-6.0",
  "interpretation": "Above normal"
}
```

**Issues:**
- ✗ No proper NHS number
- ✗ No ICD-10 code
- ✗ No BNF code
- ✗ No LOINC code
- ✗ No units or reference range

---

### Example 3: Financial Transaction Data

**What We Generate:**
```json
{
  "transaction_id": "mock_transaction_id",
  "amount": 42,
  "currency": "mock_currency",
  "date": "2025-01-01"
}
```

**What We SHOULD Generate:**
```json
{
  "transaction_id": "TXN-GB-20250115-00042",
  "amount": 1234.56,
  "currency": "GBP",
  "merchant_name": "Tesco Supermarket",
  "merchant_category": "5411",
  "merchant_category_name": "Grocery Stores, Supermarkets",
  "card_type": "debit",
  "card_last_4": "1234",
  "payment_method": "contactless",
  "date": "2025-01-15T14:32:15Z",
  "status": "completed",
  "description": "Contactless payment at Tesco Supermarket"
}
```

**Issues:**
- ✗ No proper transaction ID format
- ✗ Amount is integer not currency
- ✗ Currency is mock string
- ✗ Missing merchant information
- ✗ Missing payment details

---

## 🚀 REQUIRED FIXES FOR DATA QUALITY

### Priority 1 (Critical):

1. **Fix Type System**
   - Implement semantic type checking (age, ID, amount, etc.)
   - Ensure integers stay integers
   - Ensure proper string formatting

2. **Implement Data Validators**
   - Email validation
   - Postcode validation
   - Phone number validation
   - Date/time validation

3. **Add Cross-Field Consistency**
   - Age vs birth_date
   - Geography consistency
   - Salary vs job_title

### Priority 2 (High):

4. **Add Domain Knowledge**
   - ICD-10 codes (healthcare)
   - MCC codes (finance)
   - Product categories (e-commerce)

5. **Implement Correlation Models**
   - Age vs salary
   - Location vs income
   - Experience vs salary

6. **Improve Mock Provider**
   - Realistic defaults
   - Domain-aware generation
   - Proper type handling

### Priority 3 (Medium):

7. **Add Relationship Modeling**
   - One-to-many
   - Many-to-many
   - Hierarchical

8. **Implement Diversity Engine**
   - Diversity metrics
   - Distribution balancing
   - Geographic spread

9. **Add Anomaly Injection**
   - Outliers
   - Missing data
   - Typos
   - Duplicates

---

## 📈 SUCCESS CRITERIA

Data generation quality will be acceptable when:

✅ **100% Type Correctness** - All fields have correct types
✅ **95%+ Format Validity** - All values pass validation
✅ **90%+ Cross-Field Consistency** - Fields make sense together
✅ **80%+ Domain Accuracy** - Uses real codes/standards
✅ **85%+ Statistical Realism** - Distributions match real world
✅ **80%+ Diversity** - Good variety in generated data
✅ **0% Mock Artifacts** - No "mock_" strings in production data

---

## 💡 CONCLUSION

**Current State:** Data is generated but quality is poor (35% overall)

**Critical Issues:**
1. Type inconsistencies make data unusable
2. Mock provider generates worthless test data
3. No cross-field validation
4. No domain knowledge
5. No statistical realism

**Recommended Immediate Actions:**
1. Fix type system (integers, strings, emails)
2. Improve mock provider with realistic defaults
3. Add format validators
4. Implement basic cross-field consistency checks
5. Add domain knowledge databases

**Estimated Effort:** 1-2 weeks to get quality from 35% → 80%
