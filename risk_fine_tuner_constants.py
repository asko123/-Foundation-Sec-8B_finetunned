"""
Shared constants for risk fine-tuning and data processing.
"""

# L2 Categories (previously MACRO_RISKS)
L2 = {
    "1": "Operating Model & Risk Management",
    "2": "Develop and Acquire Software and Systems",
    "3": "Manage & Demise IT Assets",
    "4": "Manage Data",
    "5": "Protect Data",
    "6": "Identity & Access Management",
    "7": "Manage Infrastructure",
    "8": "Manage IT Vulnerabilities & Patching",
    "9": "Manage Technology Capacity & Resources",
    "10": "Monitor & Respond to Technology Incidents",
    "11": "Monitor and Respond to Security Incidents",
    "12": "Manage Business Continuity and Disaster Recovery"
}

# Macro Risks (previously THEMATIC_RISKS)
MACRO_RISKS = {
    "1": [
        "Policy/Standard Review",
        "KCI / KRI completeness",
        "IT General & Baseline Controls (Coverage)",
        "Framework Controls (External/Internal)",
        "Exception Management & Risk Tolerance",
        "Issue Management",
        "Monitoring & Testing (MAT)",
        "Security / IT Awareness Training",
        "Maturity Baseline (Yearly)",
        "Governance (Operational Controls)"
    ],
    "2": [
        "Flag Ship Control Coverage",
        "Business Requirement Approval Process",
        "Change Process (Standards & Emergency)",
        "Post Implementation Evaluation (ORE)",
        "Software Dependencies (Internal and External)",
        "M&A – Control Coverage"
    ],
    "3": [
        "Inventory Accuracy & Completeness",
        "Asset Classification & Governance",
        "End of Life – (Hardware and Software)",
        "Asset Destruction (Storage / Media)"
    ],
    "4": [
        "Data Identification, Inventory & Lineage",
        "Data Classification & Governance",
        "Data Quality Controls"
    ],
    "5": [
        "Data Monitoring Processes",
        "Encryption (At Rest, Use, Transit)",
        "Data Loss Prevention",
        "Sensitive Data Logging",
        "Third Party Data Protection",
        "Removable Media"
    ],
    "6": [
        "Authentication",
        "Authorization",
        "Privilege Management",
        "Identity Access Lifecycle (Joiners/Movers/Leavers)",
        "Segregation of Duties",
        "Secrets Management",
        "Production Access"
    ],
    "7": [
        "Configuration Management",
        "Network Segmentation",
        "Cloud Controls",
        "Data Center Management"
    ],
    "8": [
        "Scanning Completeness",
        "Patching Completeness",
        "S-SDLC drafts",
        "Vulnerability assessment and risk treatment"
    ],
    "9": [
        "Capacity Planning",
        "SLO Management",
        "Monitoring (Availability, Performance and Latency)"
    ],
    "10": [
        "Incident Identification & Classification",
        "Tech Incident Reporting & Escalation",
        "Thematic & Trends"
    ],
    "11": [
        "Incident Response Planning",
        "Incident Monitoring and Handling",
        "Security Incident Reporting & Escalation",
        "Audit Logging / Post Mortem",
        "Incident Response Testing",
        "Threat Intelligence"
    ],
    "12": [
        "Operational Resiliency",
        "Cyber Resilience"
    ]
}

# PII Protection Categories
PII_PROTECTION_CATEGORIES = {
    "PC0": "Public information with no confidentiality requirements",
    "PC1": "Internal information with basic confidentiality requirements",
    "PC3": "Confidential information with high protection requirements"
}

# Common PII Types
PII_TYPES = [
    "Name", "Email", "Phone", "Address", "SSN", "Financial", "Health", 
    "Credentials", "Biometric", "National ID", "DOB", "Gender",
    "Location", "IP Address", "Device ID", "Customer ID", "Employment"
]

# Privacy Classification Mappings
PRIVACY_CLASSIFICATIONS = {
    "PC0": "Public",
    "PC1": "Internal",
    "PC3": "Confidential"
}

# Sensitivity Levels
SENSITIVITY_LEVELS = {
    "DP10": "Public",
    "DP20": "Restricted",
    "DP30": "Highly Restricted"
}

# Keywords for risk category identification
RISK_KEYWORDS = {
    "1": ["policy", "standard", "governance", "framework", "control", "baseline", "training", "awareness", "maturity", "monitoring", "testing", "kci", "kri", "exception", "tolerance", "issue management"],
    "2": ["development", "acquire", "software", "change", "requirement", "implementation", "dependency", "m&a", "sdlc", "deployment"],
    "3": ["inventory", "asset", "classification", "end of life", "destruction", "hardware", "media", "disposal"],
    "4": ["data identification", "lineage", "data classification", "data governance", "data quality", "metadata"],
    "5": ["encryption", "data loss", "dlp", "logging", "third party", "removable media", "data protection", "at rest", "in transit"],
    "6": ["authentication", "authorization", "privilege", "access", "identity", "joiner", "mover", "leaver", "segregation", "duties", "secrets", "production"],
    "7": ["configuration", "network", "segmentation", "cloud", "data center", "infrastructure"],
    "8": ["vulnerability", "patching", "scanning", "assessment", "s-sdlc", "security testing"],
    "9": ["capacity", "planning", "slo", "availability", "performance", "latency", "monitoring"],
    "10": ["incident", "identification", "classification", "escalation", "trend", "technical"],
    "11": ["security incident", "incident response", "monitoring", "handling", "audit", "post mortem", "threat intelligence"],
    "12": ["resilience", "continuity", "disaster", "recovery", "cyber resilience", "operational"]
}

# Keywords for PII identification
PII_KEYWORDS = {
    "PC0": ["public", "marketing", "documentation", "open data", "website", "brochure"],
    "PC1": ["name", "contact", "business email", "job title", "company", "department", "customer id"],
    "PC3": ["ssn", "social security", "financial", "bank", "credit card", "health", "medical", "password", "credential", "biometric", "national id", "driver license", "passport"]
} 