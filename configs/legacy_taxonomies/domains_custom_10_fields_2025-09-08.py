"""
Journal → Field taxonomy used by dataset filtering.

- Journals can belong to MULTIPLE fields.
- Matching is case-insensitive and whitespace-normalized (no punctuation stripping).
- Unmapped journals fall into "Unclassified".

Usage:
    from src.data.domains import (
        classify_journal,
        available_fields,
        journals_for_fields,
        JOURNAL_TO_FIELDS,
        PRETTY_FIELD_NAME
    )
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Set

# ---- Normalization -----------------------------------------------------------

def _norm_title(s: str) -> str:
    # Lowercase + collapse internal whitespace
    return " ".join(s.split()).casefold().strip()


# ---- Raw field → list of journal titles -------------------------------------
# NOTE: Keep journal titles exactly as they appear in your dataset (column: source_title).
# Matching is case-insensitive, but spelling and punctuation should match.

FIELDS: Dict[str, List[str]] = {
    # 1) Entrepreneurship & Innovation
    "Entrepreneurship_and_Innovation": [
        "Research Policy",
        "Small Business Economics",
        "Journal of Open Innovation: Technology, Market, and Complexity",
        "Journal of Innovation & Knowledge",
        "Entrepreneurship Theory and Practice",
        "Journal of Business Venturing",
        "Technovation",
        "Journal of Small Business Management",
        "European Journal of Innovation Management",
        "International Journal of Entrepreneurial Behavior & Research",
        "International Entrepreneurship and Management Journal",
        "Journal of Intellectual Capital",
        "The Journal of Technology Transfer",
        "Journal of the Knowledge Economy",
        "Technology Analysis & Strategic Management",
        "Journal of Product Innovation Management",
        "Entrepreneurship & Regional Development",
        "R&D Management",
        "International Small Business Journal",
        "Journal of Business Venturing Insights",
        "Strategic Entrepreneurship Journal",
        "Family Business Review",
    ],

    # 2) Environmental & Sustainability
    "Environmental_and_Sustainability": [
        "Journal of Cleaner Production",
        "Resources, Conservation and Recycling",
        "Sustainable Cities and Society",
        "Business Ethics, Environment and Responsibility",
        "Ecological Indicators",
        "Journal of Sustainable Tourism",
    ],

    # 3) Ethics & Corporate Social Responsibility
    "Ethics_and_Corporate_Social_Responsibility": [
        "Journal of Business Ethics",
        "Corporate Social Responsibility and Environmental Management",
        "ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part B: Mechanical Engineering",
        "Digital Policy, Regulation and Governance",
        "International Journal of Disclosure and Governance",
        "Journal of Management and Governance",
        "Science and Engineering Ethics",
    ],

    # 4) Tourism
    "Tourism": [
        "Current Issues in Tourism",
        "Tourism Management",
        "Journal of Travel and Tourism Marketing",
        "European Journal of Information Systems",
        "Journal of Hospitality and Tourism Management",
        "Journal of Hospitality Marketing and Management",
        "Annals of Tourism Research",
        "Asia Pacific Journal of Tourism Research",
        "Cornell Hospitality Quarterly",
        "International Journal of Hospitality Management",
        "International Journal of Retail and Distribution Management",
        "International Journal of Tourism Research",
        "Journal of Hospitality and Tourism Technology",
        "Research in Transportation Business and Management",
        "Tourism Economics",
        "Tourism Management Perspectives",
        "Tourism Recreation Research",
        "Tourism Review",
    ],

    # 5) Information Systems
    "Information_Systems": [
        "International Journal of Information Management",
        "Information and Management",
        "MIS Quarterly",
        "Journal of Enterprise Information Management",
        "International Journal of Information Management Data Insights",
        "International Journal of Accounting and Information Management",
        "Decision Support Systems",
        "ACM Transactions on Information Systems",
        "Asia Pacific Journal of Information Systems",
        "Big Data and Society",
        "Computational Management Science",
        "Electronic Journal of Knowledge Management",
        "Enterprise Information Systems",
        "Global Journal of Flexible Systems Management",
        "IMA Journal of Management Mathematics",
        "Industrial Management and Data Systems",
        "Information and Organization",
        "Information Processing and Management",
        "Information Sciences",
        "Information Systems Research",
        "INFORMS Journal on Computing",
        "International Journal of Accounting Information Systems",
        "International Journal of Applied Decision Sciences",
        "International Journal of Information Systems and Project Management",
        "International Journal of Knowledge Management",
        "International Journal of Knowledge Management Studies",
        "International Journal of Management Science and Engineering Management",
        "Internet of Things (Netherlands)",
        "Journal of Big Data",
        "Journal of Industrial Information Integration",
        "Journal of Information Systems",
        "Journal of Information Technology",
        "Journal of Informetrics",
        "Journal of the Association for Information Science and Technology",
        "Knowledge-Based Systems",
        "Telecommunications Policy",
    ],

    # 6) Strategy & Management
    "Strategy_and_Management": [
        "Strategic Management Journal",
        "Business Strategy and the Environment",
        "Long Range Planning",
        "Journal of Business Research",
        "Journal of Business Venturing",
        "Journal of Business and Industrial Marketing",
        "Journal of Strategic Information Systems",
        "Journal of Industrial Integration and Management",
        "International Business Review",
        "Benchmarking",
        "Accounting and Business Research",
        "Global Journal of Flexible Systems Management",
        "IMA Journal of Management Mathematics",
        "Industrial Management and Data Systems",
        "International Journal of Applied Decision Sciences",
        "International Journal of Construction Management",
        "International Journal of Lean Six Sigma",
        "International Journal of Management Science and Engineering Management",
        "International Journal of Process Management and Benchmarking",
        "International Journal of Productivity and Performance Management",
        "Journal of Business Logistics",
        "Journal of Economics and Management Strategy",
        "Journal of Management Science and Engineering",
        "Journal of Manufacturing Processes",
        "Journal of Operations Management",
        "Journal of Product and Brand Management",
        "Management Science",
        "Manufacturing and Service Operations Management",
        "MIS Quarterly: Management Information Systems",
        "New Technology, Work and Employment",
        "Project Management Journal",
        "Research in Transportation Business and Management",
        "Service Business",
        "Service Industries Journal",
    ],

    # 7) Supply Chain & Operations Management
    "Supply_Chain_Operations_Management": [
        "International Journal of Production Research",
        "Production Planning and Control",
        "International Journal of Operations and Production Management",
        "Journal of Manufacturing Technology Management",
        "International Journal of Logistics Management",
        "Supply Chain Management",
        "International Journal of Physical Distribution and Logistics Management",
        "Transportation Research Part E: Logistics and Transportation Review",
        "International Journal of Production Economics",
        "Annals of Operations Research",
        "Asia Pacific Journal of Marketing and Logistics",
        "Cleaner Logistics and Supply Chain",
        "Communications in Transportation Research",
        "Computational Management Science",
        "Computers and Operations Research",
        "EURO Journal on Transportation and Logistics",
        "European Journal of Operational Research",
        "International Journal of Logistics Research and Applications",
        "International Journal of Supply Chain Management",
        "Journal of Air Transport Management",
        "Journal of Global Optimization",
        "Journal of Purchasing and Supply Management",
        "Journal of the Operational Research Society",
        "Journal of Travel Research",
        "Operations and Supply Chain Management",
        "Operations Research",
        "Operations Research Forum",
        "Operations Research Perspectives",
        "OR Spectrum",
        "Organizational Research Methods",
        "Transportation Research Part B: Methodological",
        "Transportation Research Part C: Emerging Technologies",
        "Uncertain Supply Chain Management",
    ],

    # 8) Management (general / HR / organizations)
    "Management": [
        "Journal of Business Research",
        "Journal of Management Information Systems",
        "Journal of Management Analytics",
        "Management Decision",
        "Review of Managerial Science",
        "Journal of International Business Studies",
        "International Journal of Human Resource Management",
        "Human Resource Management Review",
        "International Journal of Contemporary Hospitality Management",
        "Human Resource Management",
        "Academy of Management Journal",
        "Academy of Management Review",
        "European Journal of Management",
        "British Journal of Management",
        "International Journal of Production Economics",
        "Management Review Quarterly",
        "Journal of Hospitality and Tourism Management",
        "Journal of Industrial Integration and Management",
        "International Business Review",
        "Journal of Hospitality Marketing and Management",
        "Electronic Markets",
        "Total Quality Management and Business Excellence",
        "Benchmarking",
        "Thunderbird International Business Review",
        "Business Process Management Journal",
        "Academy of Management Annals",
        "AMS Review",
        "Annual Review of Organizational Psychology and Organizational Behavior",
        "Asia Pacific Journal of Human Resources",
        "Asia Pacific Journal of Management",
        "Asia Pacific Management Review",
        "Bottom Line",
        "Business and Society",
        "Business Horizons",
        "Central European Management Journal",
        "Computational Management Science",
        "Engineering, Construction and Architectural Management",
        "European Management Journal",
        "European Research on Management and Business Economics",
        "Global Journal of Flexible Systems Management",
        "Group and Organization Management",
        "Human Resource Development International",
        "Human Resource Development Review",
        "Human Resource Management Journal",
        "Humanities and Social Sciences Communications",
        "IMA Journal of Management Mathematics",
        "Information and Organization",
        "International Journal of Electronic Commerce",
        "International Journal of Forecasting",
        "International Journal of Health Geographics",
        "International Journal of Hospitality Management",
        "International Journal of Management and Enterprise Development",
        "International Journal of Management Education",
        "International Journal of Management Practice",
        "International Journal of Service Science, Management, Engineering, and Technology",
        "Iranian journal of Management Studies",
        "Journal of Business and Psychology",
        "Journal of Global Optimization",
        "Journal of Industrial and Business Economics",
        "Journal of Intellectual Capital",
        "Journal of International Management",
        "Journal of Management Control",
        "Journal of Management History",
        "Journal of Management Studies",
        "Journal of Management World",
        "Journal of Organizational Behavior",
        "Journal of Personal Selling and Sales Management",
        "Journal of Theoretical and Applied Electronic Commerce Research",
        "Leadership Quarterly",
        "Management Research Review",
        "Multinational Business Review",
        "Personnel Review",
        "Project Management Journal",
        "Resources Policy",
        "South Asian Journal of Human Resources Management",
        "TQM Journal",
    ],

    # 9) Finance
    "Finance": [
        "Accounting and Finance",
        "Journal of Risk and Financial Management",
        "Journal of Corporate Finance Research",
        "European Financial Management",
        "Journal of Financial Economics",
        "Emerging Markets Review",
        "Journal of Empirical Finance",
        "International Review of Financial Analysis",
        "Quantitative Finance",
        "Review of Accounting Studies",
        "Journal of International Financial Markets, Institutions and Money",
        "Journal of Finance and Data Science",
        "Mathematical Finance",
        "Contemporary Accounting Research",
        "Journal of Accounting Research",
        "International Business Review",
        "International Journal of Accounting and Information Management",
        "Accounting Horizons",
        "Accounting and Business Research",
        "Accounting, Auditing and Accountability Journal",
        "ASTIN Bulletin",
        "Borsa Istanbul Review",
        "British Accounting Review",
        "Business and Society",
        "Economic Modelling",
        "Electronic Commerce Research",
        "Energy Economics",
        "Emerging Markets Finance and Trade",
        "Equilibrium. Quarterly Journal of Economics and Economic Policy",
        "European Accounting Review",
        "European Journal of Finance",
        "European Research on Management and Business Economics",
        "Financial Innovation",
        "Foundations and Trends in Finance",
        "Global Journal of Flexible Systems Management",
        "HSE Economic Journal",
        "International Journal of Accounting Information Systems",
        "International Journal of Applied Decision Sciences",
        "International Journal of Economics and Business Research",
        "International Journal of Electronic Commerce",
        "International Journal of Forecasting",
        "International Journal of Productivity and Performance Management",
        "International Review of Economics and Finance",
        "Internet Research",
        "Journal of Applied Accounting Research",
        "Journal of Behavioral and Experimental Finance",
        "Journal of Business Economics",
        "Journal of Business Finance and Accounting",
        "Journal of Corporate Accounting and Finance",
        "Journal of Econometrics",
        "Journal of Economic Asymmetries",
        "Journal of Economic Behavior and Organization",
        "Journal of Economic Dynamics and Control",
        "Journal of Economics and Management Strategy",
        "Journal of Financial Management, Markets and Institutions",
        "Journal of Industrial and Business Economics",
        "Managerial Auditing Journal",
        "Maritime Economics and Logistics",
        "Meditari Accountancy Research",
        "Research in International Business and Finance",
        "Resources Policy",
        "Review of Quantitative Finance and Accounting",
        "Scandinavian Economic History Review",
    ],

    # 10) Marketing
    "Marketing": [
        "Journal of Marketing",
        "Journal of the Academy of Marketing Science",
        "Industrial Marketing Management",
        "Journal of Business Research",
        "Psychology and Marketing",
        "Journal of Research in Interactive Marketing",
        "International Journal of Research in Marketing",
        "Journal of Services Marketing",
        "Journal of Interactive Marketing",
        "Australasian Marketing Journal",
        "International Marketing Review",
        "Journal of Service Management",
        "Journal of Brand Management",
        "Marketing Science",
        "Journal of Consumer Marketing",
        "Journal of Retailing and Consumer Services",
        "Journal of Service Theory and Practice",
        "Journal of Marketing Theory and Practice",
        "Journal of Business and Industrial Marketing",
        "Journal of Marketing Research",
        "European Journal of Marketing",
        "Journal of Retailing",
        "Journal of Service Research",
        "Journal of Marketing Analytics",
        "Journal of Travel and Tourism Marketing",
        "Journal of Marketing Channels",
        "International Journal of Consumer Studies",
        "Journal of Hospitality Marketing and Management",
        "Electronic Markets",
        "Business Research",
        "Asia Pacific Journal of Marketing and Logistics",
        "Business Horizons",
        "Electronic Commerce Research and Applications",
        "International Journal of Retail and Distribution Management",
        "Journal of Advertising",
        "Journal of Consumer Affairs",
        "Journal of Destination Marketing and Management",
        "Journal of Global Scholars of Marketing Science: Bridging Asia and the World",
        "Journal of Interactive Advertising",
        "Journal of Personal Selling and Sales Management",
        "Journal of Product and Brand Management",
        "Journal of Strategic Marketing",
        "Marketing Letters",
        "Spanish Journal of Marketing - ESIC",
    ],
}

# ---- Build journal → fields (many-to-many) -----------------------------------

JOURNAL_TO_FIELDS: Dict[str, Set[str]] = {}
for field, journals in FIELDS.items():
    for j in journals:
        key = _norm_title(j)
        JOURNAL_TO_FIELDS.setdefault(key, set()).add(field)

ALL_FIELDS: List[str] = sorted(FIELDS.keys())

# Optional display names ( prettier labels in UI)
PRETTY_FIELD_NAME: Dict[str, str] = {
    "Entrepreneurship_and_Innovation": "Entrepreneurship & Innovation",
    "Environmental_and_Sustainability": "Environmental & Sustainability",
    "Ethics_and_Corporate_Social_Responsibility": "Ethics & CSR",
    "Tourism": "Tourism",
    "Information_Systems": "Information Systems",
    "Strategy_and_Management": "Strategy & Management",
    "Supply_Chain_Operations_Management": "Supply Chain & Operations",
    "Management": "Management",
    "Finance": "Finance",
    "Marketing": "Marketing",
    "Unclassified": "Unclassified",
}

# ---- Public helpers -----------------------------------------------------------

def classify_journal(journal_name: str) -> List[str]:
    """
    Return the list of fields for a journal name (case-insensitive).
    Returns ["Unclassified"] if not found.
    """
    if not journal_name:
        return ["Unclassified"]
    fields = JOURNAL_TO_FIELDS.get(_norm_title(journal_name))
    return sorted(fields) if fields else ["Unclassified"]


def available_fields(journal_names: Iterable[str]) -> List[str]:
    """
    Given an iterable of journal names (e.g., df['source_title']),
    return the sorted list of field names present in that set.
    """
    seen: Set[str] = set()
    for j in journal_names:
        for f in classify_journal(j):
            seen.add(f)
    return sorted(seen)


def journals_for_fields(selected_fields: Iterable[str], journal_names: Iterable[str]) -> Set[str]:
    """
    From all journal_names in the current dataset, return the subset whose
    field classification intersects selected_fields.

    selected_fields are the internal field keys (e.g., "Marketing").
    """
    wanted = {f for f in selected_fields if f}  # normalize iterable
    out: Set[str] = set()
    for j in journal_names:
        fs = set(classify_journal(j))
        if fs & wanted:
            out.add(j)
    return out