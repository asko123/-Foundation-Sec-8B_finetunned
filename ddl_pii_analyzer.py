#!/usr/bin/env python3
"""
DDL PII Analyzer - Analyzes Data Definition Language (DDL) statements to identify
potential PII data based on column names, data types, and constraints.

This tool helps with:
- Database design privacy reviews
- Automated compliance checking  
- Privacy impact assessments
- Data governance workflows
"""

import re
import json
from typing import Dict, List, Tuple, Set
from risk_fine_tuner_constants import PII_TYPES, PII_PROTECTION_CATEGORIES

class DDLPIIAnalyzer:
    """Analyzes DDL statements for potential PII data."""
    
    def __init__(self):
        # Enhanced PII column name patterns
        self.pii_column_patterns = {
            # PC3 - Confidential (High Risk)
            "PC3": {
                "SSN": [
                    r"ssn", r"social_security", r"social_security_number", r"social_sec",
                    r"tax_id", r"taxpayer_id", r"national_id", r"national_identifier"
                ],
                "Financial": [
                    r"credit_card", r"cc_number", r"card_number", r"account_number", r"bank_account",
                    r"routing_number", r"iban", r"swift", r"salary", r"income", r"wage"
                ],
                "Health": [
                    r"medical_record", r"health_id", r"patient_id", r"diagnosis", r"medication",
                    r"medical_condition", r"health_info", r"patient_number"
                ],
                "Credentials": [
                    r"password", r"passwd", r"pwd", r"secret", r"private_key", r"api_key",
                    r"token", r"auth_token", r"session_key"
                ],
                "Biometric": [
                    r"fingerprint", r"biometric", r"facial_recognition", r"iris_scan",
                    r"voice_print", r"dna", r"retina"
                ]
            },
            # PC1 - Internal (Medium Risk)  
            "PC1": {
                "Name": [
                    r"first_name", r"last_name", r"full_name", r"name", r"fname", r"lname",
                    r"given_name", r"family_name", r"surname", r"middle_name", r"nickname"
                ],
                "Email": [
                    r"email", r"email_address", r"mail", r"e_mail", r"business_email",
                    r"work_email", r"contact_email"
                ],
                "Phone": [
                    r"phone", r"telephone", r"phone_number", r"mobile", r"cell", r"contact_number",
                    r"work_phone", r"home_phone", r"mobile_number"
                ],
                "Address": [
                    r"address", r"street", r"city", r"state", r"zip", r"postal_code",
                    r"country", r"location", r"residence", r"home_address"
                ],
                "Employment": [
                    r"employee_id", r"emp_id", r"staff_id", r"job_title", r"position",
                    r"department", r"manager", r"supervisor", r"employee_number"
                ],
                "Customer ID": [
                    r"customer_id", r"client_id", r"user_id", r"member_id", r"account_id",
                    r"customer_number", r"client_number"
                ]
            },
            # PC0 - Public (Low Risk)
            "PC0": {
                "Public Info": [
                    r"company_name", r"business_name", r"public_id", r"product_id",
                    r"category", r"description", r"status", r"type", r"public_data"
                ]
            }
        }
        
        # Ambiguous patterns that could be PII or non-PII (require human review)
        self.ambiguous_patterns = {
            "name": {
                "examples": ["name", "title", "label"],
                "pii_possibility": "Could be person name (PC1) or product/company name (PC0)",
                "context_clues": "Check if related to users/customers vs products/categories"
            },
            "id": {
                "examples": ["id", "identifier", "ref", "reference"],
                "pii_possibility": "Could be customer/user ID (PC1) or system/product ID (PC0)", 
                "context_clues": "Check table context - user tables vs system tables"
            },
            "number": {
                "examples": ["number", "num", "no"],
                "pii_possibility": "Could be phone number (PC1) or order/product number (PC0)",
                "context_clues": "Check data type - VARCHAR(15) suggests phone, INT suggests system number"
            },
            "address": {
                "examples": ["address", "addr", "location"],
                "pii_possibility": "Could be personal address (PC1) or business/IP address (PC0/PC1)",
                "context_clues": "Check if customer/employee table vs system/network table"
            },
            "code": {
                "examples": ["code", "cd", "key"],
                "pii_possibility": "Could be personal code (PC1) or system code (PC0)",
                "context_clues": "Check if user-specific vs system-wide codes"
            },
            "info": {
                "examples": ["info", "data", "details", "description"],
                "pii_possibility": "Could contain personal info (PC1/PC3) or public info (PC0)",
                "context_clues": "Check column context and associated table purpose"
            }
        }
        
        # Data type patterns that suggest PII
        self.suspicious_data_types = {
            "SSN": [r"CHAR\(11\)", r"VARCHAR\(11\)", r"CHAR\(9\)", r"VARCHAR\(9\)"],  # SSN with/without dashes
            "Phone": [r"CHAR\(10\)", r"VARCHAR\(15\)", r"CHAR\(12\)"],  # Phone numbers
            "Credit Card": [r"CHAR\(16\)", r"VARCHAR\(19\)", r"CHAR\(15\)"],  # Credit card numbers
            "Email": [r"VARCHAR\(255\)", r"TEXT"],  # Email addresses
            "Name": [r"VARCHAR\(50\)", r"VARCHAR\(100\)", r"CHAR\(50\)"]  # Names
        }

    def analyze_ddl_statement(self, ddl: str) -> Dict:
        """
        Analyze a DDL statement for potential PII.
        
        Args:
            ddl: DDL statement (CREATE TABLE, ALTER TABLE, etc.)
            
        Returns:
            Analysis results with PII findings
        """
        result = {
            "statement_type": self._get_statement_type(ddl),
            "table_name": self._extract_table_name(ddl),
            "columns": [],
            "pii_summary": {
                "PC3": [],  # Confidential
                "PC1": [],  # Internal  
                "PC0": [],  # Public
                "AMBIGUOUS": []  # Requires human review
            },
            "overall_classification": "PC0",
            "ambiguous_columns": [],
            "requires_human_review": False,
            "privacy_recommendations": [],
            "compliance_flags": []
        }
        
        # Extract columns from DDL
        columns = self._extract_columns(ddl)
        
        for col_name, col_type, col_constraints in columns:
            col_analysis = self._analyze_column(col_name, col_type, col_constraints)
            result["columns"].append(col_analysis)
            
            # Add to PII summary
            if col_analysis["pii_types"]:
                pc_level = col_analysis["pc_category"]
                result["pii_summary"][pc_level].extend(col_analysis["pii_types"])
            
            # Track ambiguous columns
            if col_analysis["is_ambiguous"]:
                result["ambiguous_columns"].append({
                    "column_name": col_name,
                    "ambiguous_type": col_analysis["ambiguous_type"],
                    "pii_possibility": col_analysis["pii_possibility"],
                    "context_clues": col_analysis["context_clues"],
                    "data_type": col_type
                })
                result["pii_summary"]["AMBIGUOUS"].extend(col_analysis["pii_types"])
                result["requires_human_review"] = True
        
        # Determine overall classification using "highest sensitivity wins" + ambiguous handling
        result["overall_classification"] = self._determine_overall_classification(result["pii_summary"], result["requires_human_review"])
        
        # Generate recommendations
        result["privacy_recommendations"] = self._generate_recommendations(result)
        
        # Check compliance flags
        result["compliance_flags"] = self._check_compliance_flags(result)
        
        return result

    def _get_statement_type(self, ddl: str) -> str:
        """Extract the type of DDL statement."""
        ddl_upper = ddl.upper().strip()
        if ddl_upper.startswith("CREATE TABLE"):
            return "CREATE_TABLE"
        elif ddl_upper.startswith("ALTER TABLE"):
            return "ALTER_TABLE"
        elif ddl_upper.startswith("CREATE VIEW"):
            return "CREATE_VIEW"
        else:
            return "UNKNOWN"

    def _extract_table_name(self, ddl: str) -> str:
        """Extract table name from DDL."""
        patterns = [
            r"CREATE\s+TABLE\s+([`\"']?)(\w+)\1",
            r"ALTER\s+TABLE\s+([`\"']?)(\w+)\1",
            r"CREATE\s+VIEW\s+([`\"']?)(\w+)\1"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ddl, re.IGNORECASE)
            if match:
                return match.group(2)
        
        return "unknown_table"

    def _extract_columns(self, ddl: str) -> List[Tuple[str, str, str]]:
        """Extract column definitions from DDL."""
        columns = []
        
        # Find the column definitions section
        match = re.search(r'\((.*)\)', ddl, re.DOTALL | re.IGNORECASE)
        if not match:
            return columns
        
        column_section = match.group(1)
        
        # Split by commas (handling nested parentheses)
        column_lines = self._split_column_definitions(column_section)
        
        for line in column_lines:
            line = line.strip()
            if not line or line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'KEY')):
                continue
            
            # Parse column: name, type, constraints
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('`"\'')
                col_type = parts[1]
                col_constraints = ' '.join(parts[2:]) if len(parts) > 2 else ''
                columns.append((col_name, col_type, col_constraints))
        
        return columns

    def _split_column_definitions(self, text: str) -> List[str]:
        """Split column definitions handling nested parentheses."""
        lines = []
        current_line = ""
        paren_count = 0
        
        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                if current_line.strip():
                    lines.append(current_line.strip())
                current_line = ""
                continue
            
            current_line += char
        
        if current_line.strip():
            lines.append(current_line.strip())
        
        return lines

    def _analyze_column(self, col_name: str, col_type: str, col_constraints: str) -> Dict:
        """Analyze a single column for PII indicators."""
        result = {
            "column_name": col_name,
            "data_type": col_type,
            "constraints": col_constraints,
            "pii_types": [],
            "pc_category": "PC0",
            "confidence_score": 0.0,
            "reasoning": [],
            "is_ambiguous": False,
            "ambiguous_type": None,
            "pii_possibility": None,
            "context_clues": None
        }
        
        col_name_lower = col_name.lower()
        col_type_upper = col_type.upper()
        
        # Check column name patterns
        max_confidence = 0.0
        detected_category = "PC0"
        
        for pc_level in ["PC3", "PC1", "PC0"]:
            for pii_type, patterns in self.pii_column_patterns[pc_level].items():
                for pattern in patterns:
                    if re.search(pattern, col_name_lower):
                        confidence = 0.9  # High confidence for exact pattern match
                        if confidence > max_confidence:
                            max_confidence = confidence
                            detected_category = pc_level
                            result["pii_types"] = [pii_type]
                            result["reasoning"].append(f"Column name '{col_name}' matches {pii_type} pattern")
        
        # Check data type hints
        for pii_type, type_patterns in self.suspicious_data_types.items():
            for pattern in type_patterns:
                if re.search(pattern, col_type_upper):
                    type_confidence = 0.6  # Medium confidence for data type match
                    if type_confidence > max_confidence * 0.8:  # Don't override strong name matches
                        result["reasoning"].append(f"Data type '{col_type}' suggests {pii_type}")
                        if pii_type not in result["pii_types"]:
                            result["pii_types"].append(pii_type)
        
        # Check for ambiguous patterns (only if no clear PII match found)
        if max_confidence < 0.7:  # Only consider ambiguous if confidence is low
            for ambiguous_type, pattern_info in self.ambiguous_patterns.items():
                for pattern in pattern_info["examples"]:
                    if re.search(pattern, col_name_lower):
                        result["is_ambiguous"] = True
                        result["ambiguous_type"] = ambiguous_type
                        result["pii_possibility"] = pattern_info["pii_possibility"]
                        result["context_clues"] = pattern_info["context_clues"]
                        result["pii_types"] = [f"Ambiguous_{ambiguous_type}"]
                        result["pc_category"] = "AMBIGUOUS"
                        result["confidence_score"] = 0.5  # Medium confidence for ambiguous
                        result["reasoning"].append(f"Column name '{col_name}' is ambiguous - {pattern_info['pii_possibility']}")
                        break
                    if result["is_ambiguous"]:
                        break
        
        # Set final classification (keep existing if found clear match)
        if not result["is_ambiguous"]:
            result["pc_category"] = detected_category
            result["confidence_score"] = max_confidence
        
        return result

    def _determine_overall_classification(self, pii_summary: Dict, requires_human_review: bool = False) -> str:
        """Determine overall classification using highest sensitivity wins + ambiguous handling."""
        if pii_summary["PC3"]:
            return "PC3"
        elif pii_summary["PC1"]:
            return "PC1"
        elif requires_human_review and pii_summary["AMBIGUOUS"]:
            return "AMBIGUOUS"  # Requires human review
        else:
            return "PC0"

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate privacy recommendations based on analysis."""
        recommendations = []
        
        overall = analysis["overall_classification"]
        
        if overall == "PC3":
            recommendations.extend([
                "HIGH RISK: This table contains confidential PII data",
                "Implement strong encryption at rest and in transit",
                "Apply strict access controls and authentication",
                "Ensure GDPR/CCPA compliance measures are in place",
                "Implement data loss prevention (DLP) monitoring",
                "Consider data retention policies and right to deletion"
            ])
        elif overall == "PC1":
            recommendations.extend([
                "MEDIUM RISK: This table contains internal PII data",
                "Implement appropriate access controls",
                "Consider data masking for non-production environments",
                "Document data handling procedures",
                "Monitor access patterns for unusual activity"
            ])
        elif overall == "AMBIGUOUS":
            recommendations.extend([
                "AMBIGUOUS: This table contains columns that may or may not be PII",
                "HUMAN REVIEW REQUIRED: Manual inspection needed to determine PII status",
                "Review column context and intended use to classify properly",
                "Consider table purpose: user/customer data vs system/product data",
                "Apply precautionary principle: treat as PII until confirmed otherwise"
            ])
        else:
            recommendations.append("LOW RISK: No sensitive PII detected in this table")
        
        # Specific recommendations for detected PII types
        all_pii = analysis["pii_summary"]["PC3"] + analysis["pii_summary"]["PC1"]
        if "SSN" in all_pii:
            recommendations.append("SSN detected: Implement tokenization or format-preserving encryption")
        if "Financial" in all_pii:
            recommendations.append("Financial data detected: Ensure PCI DSS compliance")
        if "Health" in all_pii:
            recommendations.append("Health data detected: Ensure HIPAA compliance")
        
        return recommendations

    def _check_compliance_flags(self, analysis: Dict) -> List[str]:
        """Check for compliance-related flags."""
        flags = []
        
        all_pii = analysis["pii_summary"]["PC3"] + analysis["pii_summary"]["PC1"]
        
        if "SSN" in all_pii:
            flags.append("REQUIRES_GDPR_REVIEW")
            flags.append("REQUIRES_CCPA_REVIEW")
        if "Health" in all_pii:
            flags.append("REQUIRES_HIPAA_REVIEW")
        if "Financial" in all_pii:
            flags.append("REQUIRES_PCI_DSS_REVIEW")
        if analysis["overall_classification"] == "PC3":
            flags.append("REQUIRES_DPO_APPROVAL")
        elif analysis["overall_classification"] == "AMBIGUOUS":
            flags.append("REQUIRES_HUMAN_REVIEW")
            flags.append("REQUIRES_PRIVACY_OFFICER_REVIEW")
        
        return flags

def analyze_ddl_example():
    """Demonstrate DDL analysis with example statements."""
    
    analyzer = DDLPIIAnalyzer()
    
    # Example DDL statements
    examples = [
        {
            "name": "Customer Table",
            "ddl": """
            CREATE TABLE customers (
                customer_id INT PRIMARY KEY,
                first_name VARCHAR(50) NOT NULL,
                last_name VARCHAR(50) NOT NULL,
                email_address VARCHAR(255),
                phone_number VARCHAR(15),
                ssn CHAR(11),
                credit_card VARCHAR(19),
                address VARCHAR(255),
                city VARCHAR(50),
                state CHAR(2),
                zip_code VARCHAR(10),
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        },
        {
            "name": "Employee Table", 
            "ddl": """
            CREATE TABLE employees (
                employee_id INT AUTO_INCREMENT PRIMARY KEY,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                work_email VARCHAR(255),
                department VARCHAR(100),
                job_title VARCHAR(100),
                salary DECIMAL(10,2),
                manager_id INT,
                hire_date DATE
            );
            """
        },
        {
            "name": "Product Catalog",
            "ddl": """
            CREATE TABLE products (
                product_id INT PRIMARY KEY,
                product_name VARCHAR(255),
                description TEXT,
                category VARCHAR(100),
                price DECIMAL(8,2),
                status ENUM('active', 'inactive')
            );
            """
        },
        {
            "name": "Ambiguous User Data", 
            "ddl": """
            CREATE TABLE user_data (
                id INT PRIMARY KEY,
                name VARCHAR(100),
                address TEXT,
                number VARCHAR(20),
                code VARCHAR(50),
                info TEXT,
                created_date TIMESTAMP
            );
            """
        }
    ]
    
    print("DDL PII ANALYSIS EXAMPLES")
    print("=" * 80)
    
    for example in examples:
        print(f"\n{example['name']}")
        print("-" * 50)
        
        analysis = analyzer.analyze_ddl_statement(example['ddl'])
        
        print(f"Table: {analysis['table_name']}")
        print(f"Overall Classification: {analysis['overall_classification']}")
        
        print(f"\nPII Summary:")
        for level in ['PC3', 'PC1', 'PC0']:
            if analysis['pii_summary'][level]:
                print(f"   {level}: {', '.join(set(analysis['pii_summary'][level]))}")
        
        print(f"\nDetected PII Columns:")
        for col in analysis['columns']:
            if col['pii_types']:
                print(f"   • {col['column_name']} ({col['data_type']}) → {', '.join(col['pii_types'])} [{col['pc_category']}]")
        
        # Show ambiguous columns if any
        if analysis.get('ambiguous_columns'):
            print(f"\nAmbiguous Columns (Require Human Review):")
            for amb_col in analysis['ambiguous_columns']:
                print(f"   • {amb_col['column_name']} ({amb_col['data_type']}) → {amb_col['ambiguous_type']}")
                print(f"     {amb_col['pii_possibility']}")
                print(f"     {amb_col['context_clues']}")
        
        if analysis.get('requires_human_review'):
            print(f"\nHUMAN REVIEW REQUIRED: This schema contains ambiguous columns")
        
        if analysis['compliance_flags']:
            print(f"\nCompliance Flags: {', '.join(analysis['compliance_flags'])}")
        
        print(f"\nRecommendations:")
        for rec in analysis['privacy_recommendations'][:3]:  # Show top 3
            print(f"   {rec}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    analyze_ddl_example() 