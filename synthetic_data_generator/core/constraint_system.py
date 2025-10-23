"""
Constraint Satisfaction System

Defines and enforces constraints on generated data:
- Field value constraints (min, max, allowed values)
- Type constraints (format validation)
- Relationship constraints (field dependencies)
- Custom rules
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re


class ConstraintType(Enum):
    """Types of constraints"""
    RANGE = "range"              # Min/max for numbers
    LENGTH = "length"            # Min/max length for strings
    FORMAT = "format"            # Regex pattern matching
    ALLOWED_VALUES = "allowed"   # Whitelist of values
    DEPENDENCY = "dependency"    # Depends on another field
    CUSTOM = "custom"            # Custom validation function


@dataclass
class Constraint:
    """Represents a single constraint"""
    field: str
    constraint_type: ConstraintType
    parameters: Dict[str, Any]
    error_message: Optional[str] = None

    def validate(self, value: Any, record: Dict[str, Any] = None) -> tuple[bool, Optional[str]]:
        """
        Validate value against constraint

        Returns:
            (is_valid, error_message)
        """
        try:
            if self.constraint_type == ConstraintType.RANGE:
                return self._validate_range(value)
            elif self.constraint_type == ConstraintType.LENGTH:
                return self._validate_length(value)
            elif self.constraint_type == ConstraintType.FORMAT:
                return self._validate_format(value)
            elif self.constraint_type == ConstraintType.ALLOWED_VALUES:
                return self._validate_allowed_values(value)
            elif self.constraint_type == ConstraintType.DEPENDENCY:
                if record is None:
                    return False, "Cannot validate dependency without full record"
                return self._validate_dependency(value, record)
            elif self.constraint_type == ConstraintType.CUSTOM:
                return self._validate_custom(value, record)
            else:
                return False, f"Unknown constraint type: {self.constraint_type}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _validate_range(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate numeric range"""
        min_val = self.parameters.get('min')
        max_val = self.parameters.get('max')

        try:
            num_val = float(value)
            if min_val is not None and num_val < min_val:
                return False, self.error_message or f"Value {value} below minimum {min_val}"
            if max_val is not None and num_val > max_val:
                return False, self.error_message or f"Value {value} above maximum {max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, f"Value {value} is not numeric"

    def _validate_length(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate string length"""
        min_len = self.parameters.get('min', 0)
        max_len = self.parameters.get('max', float('inf'))

        length = len(str(value))
        if length < min_len:
            return False, self.error_message or f"Length {length} below minimum {min_len}"
        if length > max_len:
            return False, self.error_message or f"Length {length} above maximum {max_len}"
        return True, None

    def _validate_format(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate format using regex"""
        pattern = self.parameters.get('pattern')
        if not pattern:
            return True, None

        if not re.match(pattern, str(value)):
            return False, self.error_message or f"Value '{value}' doesn't match pattern {pattern}"
        return True, None

    def _validate_allowed_values(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate against whitelist"""
        allowed = self.parameters.get('values', [])
        if value not in allowed:
            return False, self.error_message or f"Value '{value}' not in allowed values: {allowed}"
        return True, None

    def _validate_dependency(self, value: Any, record: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate field dependency"""
        depends_on = self.parameters.get('field')
        condition = self.parameters.get('condition')  # Function: (dependent_value, this_value) -> bool

        if depends_on not in record:
            return False, f"Dependent field '{depends_on}' not found in record"

        dependent_value = record[depends_on]
        if not condition(dependent_value, value):
            return False, self.error_message or f"Dependency constraint failed for {self.field}"
        return True, None

    def _validate_custom(self, value: Any, record: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate using custom function"""
        validator = self.parameters.get('validator')
        if not validator:
            return True, None

        is_valid = validator(value, record)
        if not is_valid:
            return False, self.error_message or "Custom validation failed"
        return True, None


class ConstraintSatisfactionSystem:
    """System for defining and enforcing constraints"""

    def __init__(self):
        self.constraints: Dict[str, List[Constraint]] = {}

    def add_constraint(self, constraint: Constraint):
        """Add a constraint for a field"""
        if constraint.field not in self.constraints:
            self.constraints[constraint.field] = []
        self.constraints[constraint.field].append(constraint)

    def add_range_constraint(self, field: str, min_val: float = None, max_val: float = None, error_message: str = None):
        """Add range constraint (min/max for numbers)"""
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.RANGE,
            parameters={'min': min_val, 'max': max_val},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def add_length_constraint(self, field: str, min_len: int = None, max_len: int = None, error_message: str = None):
        """Add length constraint for strings"""
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.LENGTH,
            parameters={'min': min_len, 'max': max_len},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def add_format_constraint(self, field: str, pattern: str, error_message: str = None):
        """Add format constraint (regex pattern)"""
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.FORMAT,
            parameters={'pattern': pattern},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def add_allowed_values_constraint(self, field: str, values: List[Any], error_message: str = None):
        """Add allowed values constraint (whitelist)"""
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.ALLOWED_VALUES,
            parameters={'values': values},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def add_dependency_constraint(self, field: str, depends_on: str, condition: Callable, error_message: str = None):
        """
        Add dependency constraint (field depends on another field)

        Args:
            field: Field to constrain
            depends_on: Field it depends on
            condition: Function (dependent_value, this_value) -> bool
        """
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.DEPENDENCY,
            parameters={'field': depends_on, 'condition': condition},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def add_custom_constraint(self, field: str, validator: Callable, error_message: str = None):
        """
        Add custom constraint

        Args:
            field: Field to constrain
            validator: Function (value, record) -> bool
        """
        constraint = Constraint(
            field=field,
            constraint_type=ConstraintType.CUSTOM,
            parameters={'validator': validator},
            error_message=error_message
        )
        self.add_constraint(constraint)

    def validate_field(self, field: str, value: Any, record: Dict[str, Any] = None) -> tuple[bool, List[str]]:
        """
        Validate a single field against all its constraints

        Returns:
            (is_valid, list_of_errors)
        """
        if field not in self.constraints:
            return True, []

        errors = []
        for constraint in self.constraints[field]:
            is_valid, error_msg = constraint.validate(value, record)
            if not is_valid:
                errors.append(error_msg)

        return len(errors) == 0, errors

    def validate_record(self, record: Dict[str, Any]) -> tuple[bool, Dict[str, List[str]]]:
        """
        Validate entire record against all constraints

        Returns:
            (is_valid, dict of field -> list of errors)
        """
        all_errors = {}

        for field, value in record.items():
            is_valid, errors = self.validate_field(field, value, record)
            if not is_valid:
                all_errors[field] = errors

        return len(all_errors) == 0, all_errors

    def fix_violations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix constraint violations

        Returns:
            Fixed record
        """
        fixed_record = record.copy()

        for field, constraints in self.constraints.items():
            if field not in fixed_record:
                continue

            value = fixed_record[field]

            for constraint in constraints:
                is_valid, _ = constraint.validate(value, fixed_record)
                if not is_valid:
                    # Attempt to fix
                    fixed_value = self._attempt_fix(field, value, constraint, fixed_record)
                    if fixed_value is not None:
                        fixed_record[field] = fixed_value

        return fixed_record

    def _attempt_fix(self, field: str, value: Any, constraint: Constraint, record: Dict[str, Any]) -> Optional[Any]:
        """Attempt to fix a constraint violation"""

        if constraint.constraint_type == ConstraintType.RANGE:
            # Clamp to range
            try:
                num_val = float(value)
                min_val = constraint.parameters.get('min')
                max_val = constraint.parameters.get('max')

                if min_val is not None and num_val < min_val:
                    return min_val
                if max_val is not None and num_val > max_val:
                    return max_val
            except (ValueError, TypeError):
                pass

        elif constraint.constraint_type == ConstraintType.LENGTH:
            # Truncate or pad
            min_len = constraint.parameters.get('min', 0)
            max_len = constraint.parameters.get('max')

            str_val = str(value)
            if max_len and len(str_val) > max_len:
                return str_val[:max_len]
            if len(str_val) < min_len:
                return str_val.ljust(min_len, '0')

        # Can't fix other constraint types automatically
        return None


# Pre-defined common constraints
class CommonConstraints:
    """Common constraint definitions"""

    @staticmethod
    def age_constraint() -> Constraint:
        """Age must be between 0 and 120"""
        return Constraint(
            field='age',
            constraint_type=ConstraintType.RANGE,
            parameters={'min': 0, 'max': 120},
            error_message="Age must be between 0 and 120"
        )

    @staticmethod
    def email_format_constraint() -> Constraint:
        """Email must match email format"""
        return Constraint(
            field='email',
            constraint_type=ConstraintType.FORMAT,
            parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            error_message="Invalid email format"
        )

    @staticmethod
    def uk_postcode_constraint() -> Constraint:
        """UK postcode format"""
        return Constraint(
            field='postcode',
            constraint_type=ConstraintType.FORMAT,
            parameters={'pattern': r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'},
            error_message="Invalid UK postcode format"
        )

    @staticmethod
    def uk_phone_constraint() -> Constraint:
        """UK phone number format"""
        return Constraint(
            field='phone',
            constraint_type=ConstraintType.FORMAT,
            parameters={'pattern': r'^(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}$|^(\+44\s?[1-2]\d{1,4}|\(?0[1-2]\d{1,4}\)?)\s?\d{3,4}\s?\d{3,4}$'},
            error_message="Invalid UK phone format"
        )

    @staticmethod
    def positive_amount_constraint(field: str = 'amount') -> Constraint:
        """Amount must be positive"""
        return Constraint(
            field=field,
            constraint_type=ConstraintType.RANGE,
            parameters={'min': 0},
            error_message=f"{field} must be positive"
        )

    @staticmethod
    def end_after_start_constraint() -> Constraint:
        """End date must be after start date"""
        return Constraint(
            field='end_date',
            constraint_type=ConstraintType.DEPENDENCY,
            parameters={
                'field': 'start_date',
                'condition': lambda start, end: end > start
            },
            error_message="End date must be after start date"
        )
