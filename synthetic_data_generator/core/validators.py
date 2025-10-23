"""
Data Validators

Provides validation functions for:
- Format validation (email, phone, postcode, dates)
- Cross-field validation (consistency checks)
- Type validation
- Domain-specific validation
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re


class FormatValidator:
    """Validates data formats"""

    @staticmethod
    def validate_email(email: str) -> tuple[bool, Optional[str]]:
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False, "Email must be a non-empty string"

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, f"Invalid email format: {email}"

        return True, None

    @staticmethod
    def validate_uk_postcode(postcode: str) -> tuple[bool, Optional[str]]:
        """Validate UK postcode format"""
        if not postcode or not isinstance(postcode, str):
            return False, "Postcode must be a non-empty string"

        pattern = r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'
        if not re.match(pattern, postcode.upper()):
            return False, f"Invalid UK postcode: {postcode}"

        return True, None

    @staticmethod
    def validate_uk_phone(phone: str) -> tuple[bool, Optional[str]]:
        """Validate UK phone number"""
        if not phone or not isinstance(phone, str):
            return False, "Phone must be a non-empty string"

        # UK mobile or landline
        mobile_pattern = r'^(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}$'
        landline_pattern = r'^(\+44\s?[1-2]\d{1,4}|\(?0[1-2]\d{1,4}\)?)\s?\d{3,4}\s?\d{3,4}$'

        if not (re.match(mobile_pattern, phone) or re.match(landline_pattern, phone)):
            return False, f"Invalid UK phone: {phone}"

        return True, None

    @staticmethod
    def validate_date_format(date_str: str, format: str = '%Y-%m-%d') -> tuple[bool, Optional[str]]:
        """Validate date format"""
        if not date_str or not isinstance(date_str, str):
            return False, "Date must be a non-empty string"

        try:
            datetime.strptime(date_str, format)
            return True, None
        except ValueError:
            return False, f"Invalid date format: {date_str} (expected {format})"

    @staticmethod
    def validate_uk_date_format(date_str: str) -> tuple[bool, Optional[str]]:
        """Validate UK date format (DD/MM/YYYY)"""
        return FormatValidator.validate_date_format(date_str, '%d/%m/%Y')

    @staticmethod
    def validate_currency(amount: Any) -> tuple[bool, Optional[str]]:
        """Validate currency amount"""
        try:
            val = float(str(amount).replace('£', '').replace(',', ''))
            if val < 0:
                return False, f"Currency amount cannot be negative: {amount}"
            return True, None
        except (ValueError, TypeError):
            return False, f"Invalid currency format: {amount}"


class TypeValidator:
    """Validates data types"""

    @staticmethod
    def validate_type(value: Any, expected_type: str) -> tuple[bool, Optional[str]]:
        """Validate value matches expected type"""

        type_checks = {
            'string': lambda v: isinstance(v, str),
            'number': lambda v: isinstance(v, (int, float)),
            'integer': lambda v: isinstance(v, int),
            'float': lambda v: isinstance(v, float),
            'boolean': lambda v: isinstance(v, bool),
            'list': lambda v: isinstance(v, list),
            'dict': lambda v: isinstance(v, dict),
            'date': lambda v: isinstance(v, str),  # Date strings
            'email': lambda v: isinstance(v, str),  # Email strings
        }

        checker = type_checks.get(expected_type.lower())
        if not checker:
            return False, f"Unknown type: {expected_type}"

        if not checker(value):
            return False, f"Expected {expected_type}, got {type(value).__name__}"

        return True, None

    @staticmethod
    def validate_schema(record: Dict[str, Any], schema: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate record matches schema"""
        errors = []

        # Check all schema fields are present
        for field, expected_type in schema.items():
            if field not in record:
                errors.append(f"Missing required field: {field}")
                continue

            is_valid, error = TypeValidator.validate_type(record[field], expected_type)
            if not is_valid:
                errors.append(f"Field '{field}': {error}")

        # Check for unexpected fields
        for field in record.keys():
            if field not in schema and not field.startswith('_'):
                errors.append(f"Unexpected field: {field}")

        return len(errors) == 0, errors


class CrossFieldValidator:
    """Validates consistency across multiple fields"""

    @staticmethod
    def validate_age_vs_birth_date(age: int, birth_date: str, current_year: int = 2025) -> tuple[bool, Optional[str]]:
        """Validate age matches birth date"""
        try:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                try:
                    birth = datetime.strptime(birth_date, fmt)
                    calculated_age = current_year - birth.year
                    if abs(calculated_age - age) > 1:  # Allow 1 year tolerance
                        return False, f"Age {age} doesn't match birth date {birth_date} (calculated: {calculated_age})"
                    return True, None
                except ValueError:
                    continue

            return False, f"Could not parse birth date: {birth_date}"

        except Exception as e:
            return False, f"Error validating age vs birth date: {str(e)}"

    @staticmethod
    def validate_dates_order(start_field: str, end_field: str, record: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate end date is after start date"""
        if start_field not in record or end_field not in record:
            return True, None  # Skip if fields missing

        try:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                try:
                    start = datetime.strptime(record[start_field], fmt)
                    end = datetime.strptime(record[end_field], fmt)

                    if end <= start:
                        return False, f"{end_field} must be after {start_field}"
                    return True, None
                except ValueError:
                    continue

            return False, "Could not parse dates"

        except Exception as e:
            return False, f"Error validating date order: {str(e)}"

    @staticmethod
    def validate_geography_consistency(record: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate geographic fields are consistent"""
        errors = []

        country = record.get('country', '').upper()
        postcode = record.get('postcode', '')
        state = record.get('state')

        if country == 'UK' or country == 'UNITED KINGDOM':
            # UK shouldn't have states
            if state:
                errors.append(f"UK addresses should not have state: {state}")

            # Check postcode format
            if postcode:
                is_valid, error = FormatValidator.validate_uk_postcode(postcode)
                if not is_valid:
                    errors.append(f"Invalid UK postcode: {error}")

        elif country == 'US' or country == 'USA' or country == 'UNITED STATES':
            # US should have states
            if not state:
                errors.append("US addresses should have a state")

        return len(errors) == 0, errors

    @staticmethod
    def validate_salary_vs_job_title(salary: float, job_title: str) -> tuple[bool, Optional[str]]:
        """Validate salary is reasonable for job title"""

        # Rough salary ranges by job level
        salary_ranges = {
            'intern': (15000, 30000),
            'junior': (20000, 40000),
            'mid': (35000, 65000),
            'senior': (55000, 100000),
            'lead': (70000, 120000),
            'principal': (80000, 150000),
            'manager': (50000, 90000),
            'director': (80000, 150000),
            'vp': (100000, 200000),
            'ceo': (120000, 500000),
            'executive': (100000, 300000),
        }

        job_lower = job_title.lower()

        for level, (min_sal, max_sal) in salary_ranges.items():
            if level in job_lower:
                if salary < min_sal * 0.7 or salary > max_sal * 1.5:  # 30% tolerance
                    return False, f"Salary {salary} seems unrealistic for {job_title} (expected £{min_sal}-£{max_sal})"
                return True, None

        # No match found, can't validate
        return True, None

    @staticmethod
    def validate_email_vs_name(email: str, first_name: str = None, last_name: str = None) -> tuple[bool, Optional[str]]:
        """Validate email contains name (loose check)"""
        if not email or '@' not in email:
            return False, "Invalid email format"

        local_part = email.split('@')[0].lower()

        # Check if name appears in email (very loose check)
        if first_name and first_name.lower() in local_part:
            return True, None
        if last_name and last_name.lower() in local_part:
            return True, None

        # If we have names but they don't appear, it's suspicious but not necessarily wrong
        # (could be username, etc.)
        return True, None


class DomainValidator:
    """Domain-specific validators"""

    @staticmethod
    def validate_healthcare_icd10_code(code: str) -> tuple[bool, Optional[str]]:
        """Validate ICD-10 code format"""
        # ICD-10 format: Letter followed by 2-3 digits, optionally followed by decimal and 1-2 digits
        pattern = r'^[A-Z]\d{2,3}(\.\d{1,2})?$'

        if not re.match(pattern, code):
            return False, f"Invalid ICD-10 code format: {code}"

        return True, None

    @staticmethod
    def validate_financial_transaction_type(trans_type: str) -> tuple[bool, Optional[str]]:
        """Validate financial transaction type"""
        valid_types = [
            'PURCHASE', 'REFUND', 'TRANSFER', 'PAYMENT', 'WITHDRAWAL',
            'DEPOSIT', 'FEE', 'INTEREST', 'DIVIDEND', 'ADJUSTMENT'
        ]

        if trans_type.upper() not in valid_types:
            return False, f"Invalid transaction type: {trans_type} (allowed: {valid_types})"

        return True, None

    @staticmethod
    def validate_currency_code(code: str) -> tuple[bool, Optional[str]]:
        """Validate ISO 4217 currency code"""
        # Common currency codes
        valid_codes = [
            'GBP', 'USD', 'EUR', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
            'CNY', 'INR', 'BRL', 'ZAR', 'SEK', 'NOK', 'DKK'
        ]

        if code.upper() not in valid_codes:
            return False, f"Invalid currency code: {code}"

        return True, None


class ValidationResult:
    """Result of validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.field_errors: Dict[str, List[str]] = {}

    def add_error(self, error: str, field: str = None):
        """Add an error"""
        self.is_valid = False
        self.errors.append(error)
        if field:
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].append(error)

    def add_warning(self, warning: str):
        """Add a warning (doesn't invalidate)"""
        self.warnings.append(warning)

    def __str__(self) -> str:
        if self.is_valid:
            return "✓ Valid"

        output = ["✗ Validation failed:"]
        for error in self.errors:
            output.append(f"  - {error}")

        if self.warnings:
            output.append("\nWarnings:")
            for warning in self.warnings:
                output.append(f"  - {warning}")

        return "\n".join(output)


class DataValidator:
    """Main validator that combines all validation types"""

    def __init__(self):
        self.format_validator = FormatValidator()
        self.type_validator = TypeValidator()
        self.cross_field_validator = CrossFieldValidator()
        self.domain_validator = DomainValidator()

    def validate_record(self, record: Dict[str, Any], schema: Dict[str, str] = None) -> ValidationResult:
        """
        Comprehensive validation of a record

        Args:
            record: Record to validate
            schema: Optional schema to validate against

        Returns:
            ValidationResult with all errors and warnings
        """
        result = ValidationResult()

        # Type validation
        if schema:
            is_valid, errors = self.type_validator.validate_schema(record, schema)
            if not is_valid:
                for error in errors:
                    result.add_error(error)

        # Format validation for known fields
        if 'email' in record:
            is_valid, error = self.format_validator.validate_email(record['email'])
            if not is_valid:
                result.add_error(error, 'email')

        if 'postcode' in record or 'postal_code' in record:
            postcode = record.get('postcode') or record.get('postal_code')
            is_valid, error = self.format_validator.validate_uk_postcode(postcode)
            if not is_valid:
                result.add_warning(f"Postcode format: {error}")  # Warning not error

        if 'phone' in record or 'mobile' in record:
            phone = record.get('phone') or record.get('mobile')
            is_valid, error = self.format_validator.validate_uk_phone(phone)
            if not is_valid:
                result.add_warning(f"Phone format: {error}")

        # Cross-field validation
        if 'age' in record and 'birth_date' in record:
            is_valid, error = self.cross_field_validator.validate_age_vs_birth_date(
                record['age'], record['birth_date']
            )
            if not is_valid:
                result.add_error(error)

        if 'start_date' in record and 'end_date' in record:
            is_valid, error = self.cross_field_validator.validate_dates_order(
                'start_date', 'end_date', record
            )
            if not is_valid:
                result.add_error(error)

        # Geography consistency
        is_valid, errors = self.cross_field_validator.validate_geography_consistency(record)
        if not is_valid:
            for error in errors:
                result.add_warning(error)

        # Salary vs job title
        if 'salary' in record and 'job_title' in record:
            is_valid, error = self.cross_field_validator.validate_salary_vs_job_title(
                record['salary'], record['job_title']
            )
            if not is_valid:
                result.add_warning(error)

        return result

    def validate_batch(self, records: List[Dict[str, Any]], schema: Dict[str, str] = None) -> List[ValidationResult]:
        """Validate multiple records"""
        return [self.validate_record(record, schema) for record in records]
