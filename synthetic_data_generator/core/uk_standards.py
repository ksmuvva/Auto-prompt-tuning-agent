"""
UK Standards Compliance Layer

Ensures all generated data follows UK standards:
- UK date format (DD/MM/YYYY)
- UK postcodes
- UK phone numbers
- UK currency (£)
- UK names (diverse demographics)
- GDPR compliance
"""

from typing import Dict, Any, List, Optional
import random
import re
from datetime import datetime, timedelta


class UKStandardsValidator:
    """Validator for UK standards compliance"""

    # UK Postcode patterns
    POSTCODE_PATTERN = re.compile(
        r'^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$',
        re.IGNORECASE
    )

    # UK Phone patterns
    MOBILE_PATTERN = re.compile(r'^(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}$')
    LANDLINE_PATTERN = re.compile(r'^(\+44\s?[1-2]\d{1,4}|\(?0[1-2]\d{1,4}\)?)\s?\d{3,4}\s?\d{3,4}$')

    @staticmethod
    def validate_postcode(postcode: str) -> bool:
        """Validate UK postcode format"""
        return bool(UKStandardsValidator.POSTCODE_PATTERN.match(postcode.strip()))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate UK phone number"""
        return (UKStandardsValidator.MOBILE_PATTERN.match(phone) or
                UKStandardsValidator.LANDLINE_PATTERN.match(phone))

    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate UK date format (DD/MM/YYYY)"""
        try:
            datetime.strptime(date_str, '%d/%m/%Y')
            return True
        except ValueError:
            return False


class UKStandardsGenerator:
    """Generator for UK-compliant data"""

    def __init__(self):
        self.validator = UKStandardsValidator()

    def generate_postcode(self, region: Optional[str] = None) -> str:
        """
        Generate realistic UK postcode

        Args:
            region: Optional region (e.g., 'London', 'Manchester')

        Returns:
            Valid UK postcode
        """
        # UK postcode areas
        postcode_areas = {
            'London': ['SW1A', 'EC1A', 'WC1A', 'E1', 'N1', 'SE1', 'W1'],
            'Manchester': ['M1', 'M2', 'M3', 'M4', 'M15', 'M20'],
            'Birmingham': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'Edinburgh': ['EH1', 'EH2', 'EH3', 'EH8'],
            'Cardiff': ['CF10', 'CF11', 'CF14', 'CF24'],
            'Leeds': ['LS1', 'LS2', 'LS6', 'LS11'],
            'Glasgow': ['G1', 'G2', 'G3', 'G12'],
            'Liverpool': ['L1', 'L2', 'L3', 'L8'],
            'Bristol': ['BS1', 'BS2', 'BS3', 'BS8'],
        }

        # Select area
        if region and region in postcode_areas:
            area = random.choice(postcode_areas[region])
        else:
            # Random from all areas
            all_areas = [area for areas in postcode_areas.values() for area in areas]
            area = random.choice(all_areas)

        # Generate inward code (e.g., 1AA, 2BB)
        inward = f"{random.randint(1, 9)}{random.choice('ABCDEFGHJKLMNPQRSTUVWXY')}{random.choice('ABCDEFGHJKLMNPQRSTUVWXY')}"

        postcode = f"{area} {inward}"

        assert self.validator.validate_postcode(postcode), f"Generated invalid postcode: {postcode}"
        return postcode

    def generate_phone(self, phone_type: str = 'mobile') -> str:
        """
        Generate realistic UK phone number

        Args:
            phone_type: 'mobile' or 'landline'

        Returns:
            Valid UK phone number
        """
        if phone_type == 'mobile':
            # UK mobile: 07XXX XXXXXX
            prefix = random.choice(['07700', '07800', '07900', '07400', '07500'])
            number = f"{random.randint(100000, 999999)}"
            phone = f"{prefix} {number[:3]} {number[3:]}"
        else:
            # UK landline: 020 XXXX XXXX (London) or 0161 XXX XXXX (Manchester), etc.
            area_codes = ['020', '0161', '0121', '0131', '0113', '0141']
            area = random.choice(area_codes)
            if area == '020':  # London format
                local = f"{random.randint(7000, 8999)} {random.randint(1000, 9999)}"
            else:
                local = f"{random.randint(200, 999)} {random.randint(1000, 9999)}"
            phone = f"{area} {local}"

        assert self.validator.validate_phone(phone), f"Generated invalid phone: {phone}"
        return phone

    def generate_name(self, gender: Optional[str] = None) -> tuple[str, str]:
        """
        Generate realistic UK name with diverse demographics

        Args:
            gender: Optional gender ('male', 'female', or None for random)

        Returns:
            Tuple of (first_name, last_name)
        """
        # UK ethnic diversity (realistic proportions)
        ethnicity = random.choices(
            ['british', 'asian', 'black', 'mixed', 'other'],
            weights=[80, 9, 4, 3, 4]  # Approximate UK demographics
        )[0]

        if not gender:
            gender = random.choice(['male', 'female'])

        first_names = self._get_first_names(ethnicity, gender)
        last_names = self._get_last_names(ethnicity)

        return random.choice(first_names), random.choice(last_names)

    def _get_first_names(self, ethnicity: str, gender: str) -> List[str]:
        """Get first names by ethnicity and gender"""
        names = {
            'british': {
                'male': ['Oliver', 'George', 'Harry', 'Jack', 'Jacob', 'Noah', 'Charlie', 'James', 'Thomas', 'William'],
                'female': ['Olivia', 'Amelia', 'Isla', 'Emily', 'Ava', 'Jessica', 'Poppy', 'Sophie', 'Isabella', 'Lily']
            },
            'asian': {
                'male': ['Mohammed', 'Ali', 'Hassan', 'Omar', 'Amir', 'Arjun', 'Rahul', 'Ravi', 'Wei', 'Jun'],
                'female': ['Aisha', 'Fatima', 'Zara', 'Priya', 'Ananya', 'Mei', 'Li', 'Yuki', 'Sakura', 'Hina']
            },
            'black': {
                'male': ['Tyrone', 'Marcus', 'Jamal', 'Kwame', 'Kofi', 'Emmanuel', 'David', 'Samuel', 'Joshua', 'Daniel'],
                'female': ['Ama', 'Abena', 'Grace', 'Faith', 'Joy', 'Blessing', 'Precious', 'Princess', 'Angela', 'Michelle']
            },
            'mixed': {
                'male': ['Noah', 'Mason', 'Ethan', 'Lucas', 'Liam', 'Alexander', 'James', 'Oliver', 'Benjamin', 'Henry'],
                'female': ['Maya', 'Zara', 'Layla', 'Mia', 'Ella', 'Emma', 'Sophia', 'Ava', 'Chloe', 'Ruby']
            },
            'other': {
                'male': ['Adam', 'Daniel', 'Michael', 'Alexander', 'Benjamin', 'Samuel', 'Joseph', 'David', 'Matthew', 'Andrew'],
                'female': ['Sarah', 'Rebecca', 'Rachel', 'Hannah', 'Maria', 'Anna', 'Eva', 'Sophia', 'Elena', 'Nina']
            }
        }
        return names.get(ethnicity, names['british']).get(gender, names['british']['male'])

    def _get_last_names(self, ethnicity: str) -> List[str]:
        """Get last names by ethnicity"""
        surnames = {
            'british': ['Smith', 'Jones', 'Taylor', 'Brown', 'Williams', 'Wilson', 'Johnson', 'Davies', 'Robinson', "O'Brien"],
            'asian': ['Khan', 'Ali', 'Ahmed', 'Patel', 'Shah', 'Chen', 'Wong', 'Lee', 'Kim', 'Singh'],
            'black': ['Okonkwo', 'Mensah', 'Okeke', 'Adeyemi', 'Johnson', 'Williams', 'Thompson', 'Campbell', 'Clarke', 'Jackson'],
            'mixed': ['Thompson', 'Garcia', 'Martinez', 'Rodriguez', 'Lopez', 'Santos', 'Silva', 'Costa', 'Fernandes', 'Pereira'],
            'other': ['Kowalski', 'Nowak', 'Dubois', 'Rossi', 'Müller', 'Schmidt', 'Petrov', 'Ivanov', 'Nielsen', 'Andersen']
        }
        return surnames.get(ethnicity, surnames['british'])

    def generate_email(self, first_name: str, last_name: str) -> str:
        """
        Generate realistic UK email address

        Args:
            first_name: Person's first name
            last_name: Person's last name

        Returns:
            Email address
        """
        # UK email providers
        providers = ['gmail.com', 'yahoo.co.uk', 'hotmail.co.uk', 'outlook.com',
                    'btinternet.com', 'sky.com', 'virginmedia.com']

        # Email formats
        formats = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name[0].lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}",
        ]

        email_format = random.choice(formats)
        provider = random.choice(providers)

        return f"{email_format}@{provider}"

    def format_date_uk(self, date: datetime) -> str:
        """
        Format date in UK format (DD/MM/YYYY)

        Args:
            date: Date to format

        Returns:
            UK-formatted date string
        """
        return date.strftime('%d/%m/%Y')

    def generate_random_date(self, start_year: int = 2020, end_year: int = 2025) -> str:
        """
        Generate random date in UK format

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            UK-formatted date string
        """
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)

        return self.format_date_uk(random_date)

    def format_currency(self, amount: float) -> str:
        """
        Format currency in UK format (£X,XXX.XX)

        Args:
            amount: Amount to format

        Returns:
            UK-formatted currency string
        """
        return f"£{amount:,.2f}"

    def generate_address(self, region: Optional[str] = None) -> Dict[str, str]:
        """
        Generate realistic UK address

        Args:
            region: Optional region for postcode

        Returns:
            Dictionary with address components
        """
        # UK street names
        street_names = [
            'High Street', 'Station Road', 'Church Lane', 'Main Street', 'Park Road',
            'Victoria Road', 'Albert Street', 'King Street', 'Queen Street', 'Manor Road'
        ]

        # UK cities
        cities = [
            'London', 'Manchester', 'Birmingham', 'Edinburgh', 'Cardiff',
            'Leeds', 'Glasgow', 'Liverpool', 'Bristol', 'Newcastle'
        ]

        house_number = random.randint(1, 150)
        street = random.choice(street_names)
        city = region if region and region in cities else random.choice(cities)
        postcode = self.generate_postcode(city)

        return {
            'street_address': f"{house_number} {street}",
            'city': city,
            'postcode': postcode,
            'country': 'United Kingdom'
        }

    def add_gdpr_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add GDPR compliance metadata

        Args:
            data: Generated data

        Returns:
            Data with GDPR metadata
        """
        data['_metadata'] = {
            'data_type': 'synthetic',
            'generated_date': datetime.now().isoformat(),
            'gdpr_compliant': True,
            'contains_pii': False,
            'purpose': 'testing/development',
            'disclaimer': 'This is synthetic data. Any resemblance to real persons is coincidental.'
        }
        return data


class UKStandardsEnforcer:
    """Enforces UK standards on generated data"""

    def __init__(self):
        self.generator = UKStandardsGenerator()
        self.validator = UKStandardsValidator()

    def enforce_standards(self, data: Dict[str, Any], schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Enforce UK standards on generated data

        Args:
            data: Generated data
            schema: Data schema

        Returns:
            Data with UK standards enforced
        """
        for field, value in data.items():
            field_type = schema.get(field, '').lower()

            # Fix postcodes
            if 'postcode' in field.lower() or 'postal' in field.lower():
                if not self.validator.validate_postcode(str(value)):
                    data[field] = self.generator.generate_postcode()

            # Fix phone numbers
            elif 'phone' in field.lower() or 'mobile' in field.lower():
                if not self.validator.validate_phone(str(value)):
                    phone_type = 'mobile' if 'mobile' in field.lower() else 'landline'
                    data[field] = self.generator.generate_phone(phone_type)

            # Fix dates
            elif field_type == 'date' or 'date' in field.lower():
                if not self.validator.validate_date_format(str(value)):
                    data[field] = self.generator.generate_random_date()

            # Fix currency
            elif 'price' in field.lower() or 'amount' in field.lower() or 'cost' in field.lower():
                try:
                    amount = float(str(value).replace('£', '').replace(',', ''))
                    data[field] = self.generator.format_currency(amount)
                except ValueError:
                    pass

        return data
