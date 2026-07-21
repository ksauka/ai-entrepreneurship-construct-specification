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
    # ---- Custom fields (based on ASJC 1400-series + (Google Scholar style)) -------
    # ─────────────────────────────────────────────────────────────
    # Custom domain  (Google Scholar style)
    # ─────────────────────────────────────────────────────────────
    "Entrepreneurship_and_Innovation": [
        "Entrepreneurship Theory and Practice",
        "Journal of Business Venturing",
        "Journal of Business Venturing Insights",
        "Strategic Entrepreneurship Journal",
        "Small Business Economics",
        "International Small Business Journal",
        "Journal of Small Business Management",
        "International Entrepreneurship and Management Journal",
        "International Journal of Entrepreneurial Behavior & Research",
        "Entrepreneurship & Regional Development",
        "Journal of Product Innovation Management",
        "Research Policy",
        "Technovation",
        "R&D Management",
        "The Journal of Technology Transfer",
        "European Journal of Innovation Management",
        "Journal of Innovation & Knowledge",
        "Journal of Intellectual Capital",
        "Journal of the Knowledge Economy",
        "Technology Analysis & Strategic Management",
        "Family Business Review",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1408 — Strategy and Management
    # ─────────────────────────────────────────────────────────────
    "Strategy_and_Management": [
        "Strategic Management Journal",
        "Long Range Planning",
        "Journal of Management Studies",
        "Journal of Business Research",
        "British Journal of Management",
        "European Management Journal",
        "Management Review Quarterly",
        "Business Horizons",
        "Business & Society",
        "Project Management Journal",
        "Service Industries Journal",
        "Benchmarking: An International Journal",
        "Business Process Management Journal",
        "Group & Organization Management",
        "Leadership Quarterly",
        "Academy of Management Annals",
        "Review of Managerial Science",
        "Total Quality Management & Business Excellence",
        "Journal of Business Logistics",
        "Journal of Industrial and Business Economics",
        "New Technology, Work and Employment",
        "Thunderbird International Business Review",
        "AMS Review",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1404 — Management Information Systems
    # ─────────────────────────────────────────────────────────────
    "Management_Information_Systems": [
        "MIS Quarterly",
        "Information Systems Research",
        "Journal of Management Information Systems",
        "Journal of Information Technology",
        "Information & Management",
        "International Journal of Information Management",
        "Information Processing & Management",
        "Decision Support Systems",
        "Journal of Strategic Information Systems",
        "Electronic Markets",
        "Internet Research",
        "Journal of Enterprise Information Management",
        "Industrial Management & Data Systems",
        "Enterprise Information Systems",
        "International Journal of Information Management Data Insights",
        "International Journal of Information Systems and Project Management",
        "International Journal of Accounting Information Systems",
        "ACM Transactions on Information Systems",
        "INFORMS Journal on Computing",
        "Information and Organization",
        "Journal of Information Systems",
        "Journal of Informetrics",
        "International Journal of Accounting and Information Management",
        "Computational Management Science",
        "International Journal of Knowledge Management",
        "International Journal of Knowledge Management Studies",
        "International Journal of Management Science and Engineering Management",
        "Journal of Industrial Information Integration",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1406 — Marketing
    # ─────────────────────────────────────────────────────────────
    "Marketing": [
        "Journal of Marketing",
        "Journal of Marketing Research",
        "Journal of the Academy of Marketing Science",
        "Marketing Science",
        "European Journal of Marketing",
        "Industrial Marketing Management",
        "International Journal of Research in Marketing",
        "Journal of Retailing",
        "Journal of Retailing and Consumer Services",
        "Journal of Service Research",
        "Journal of Service Management",
        "Journal of Services Marketing",
        "Journal of Interactive Marketing",
        "Journal of Research in Interactive Marketing",
        "Journal of Advertising",
        "Journal of Personal Selling & Sales Management",
        "Journal of Product & Brand Management",
        "Journal of Strategic Marketing",
        "Journal of Marketing Theory and Practice",
        "Psychology & Marketing",
        "International Marketing Review",
        "Australasian Marketing Journal",
        "International Journal of Consumer Studies",
        "Journal of Marketing Analytics",
        "Spanish Journal of Marketing - ESIC",
        "Asia Pacific Journal of Marketing and Logistics",
        "Electronic Commerce Research and Applications",
        "Business Research",
        "Electronic Markets",
        "Journal of Global Scholars of Marketing Science: Bridging Asia and the World",
        "Journal of Interactive Advertising",
        "Journal of Travel & Tourism Marketing",
        "Journal of Hospitality Marketing & Management",
        "International Journal of Retail & Distribution Management",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1407 — Organizational Behavior and Human Resource Management
    # ─────────────────────────────────────────────────────────────
    "Organizational_Behavior_and_Human_Resource_Management": [
        "Journal of Organizational Behavior",
        "International Journal of Human Resource Management",
        "Human Resource Management",
        "Human Resource Management Journal",
        "Human Resource Management Review",
        "Asia Pacific Journal of Human Resources",
        "Journal of Business and Psychology",
        "Personnel Review",
        "Management Decision",
        "Annual Review of Organizational Psychology and Organizational Behavior",
        "Human Resource Development International",
        "Human Resource Development Review",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1403 — Business and International Management
    # ─────────────────────────────────────────────────────────────
    "Business_and_International_Management": [
        "International Business Review",
        "Journal of International Business Studies",
        "Journal of International Management",
        "Multinational Business Review",
        "International Journal of Management Education",
        "International Journal of Management Practice",
        "International Journal of Management and Enterprise Development",
        "Central European Management Journal",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1405 — Management of Technology and Innovation
    # (innovation/tech-strategy titles that are Business-coded)
    # ─────────────────────────────────────────────────────────────
    "Technology_and_Innovation_Management": [
        "R&D Management",
        "Technology Analysis & Strategic Management",
        "Technovation",
        "The Journal of Technology Transfer",
        "European Journal of Innovation Management",
        "Journal of Innovation & Knowledge",
        "Journal of Product Innovation Management",
        "Journal of Intellectual Capital",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1402 — Accounting
    # ─────────────────────────────────────────────────────────────
    "Accounting": [
        "Accounting, Organizations and Society",
        "The Accounting Review",
        "Journal of Accounting Research",
        "Review of Accounting Studies",
        "Contemporary Accounting Research",
        "European Accounting Review",
        "Accounting and Business Research",
        "Accounting Horizons",
        "Accounting and Finance",
        "British Accounting Review",
        "Managerial Auditing Journal",
        "Journal of Applied Accounting Research",
        "Meditari Accountancy Research",
        "International Journal of Accounting Information Systems",
        "International Journal of Accounting and Information Management",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1409 — Tourism, Leisure and Hospitality Management
    # ─────────────────────────────────────────────────────────────
    "Tourism_Leisure_Hospitality": [
        "Annals of Tourism Research",
        "Tourism Management",
        "Tourism Management Perspectives",
        "Tourism Economics",
        "Tourism Review",
        "Tourism Recreation Research",
        "Current Issues in Tourism",
        "Asia Pacific Journal of Tourism Research",
        "Journal of Travel Research",
        "International Journal of Tourism Research",
        "International Journal of Hospitality Management",
        "Journal of Hospitality and Tourism Management",
        "Cornell Hospitality Quarterly",
        "Journal of Hospitality and Tourism Technology",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1410 — Industrial Relations
    # (kept minimal; few from your list fit cleanly here)
    # ─────────────────────────────────────────────────────────────
    "Industrial_Relations": [
        # Add IR titles you use (e.g., Industrial Relations Journal) if needed
    ],

    # ─────────────────────────────────────────────────────────────
    # Finance cluster (spans 1402 Accounting + 2000-series Economics in Scopus)
    # Here we keep business-side finance and accounting journals you listed.
    # ─────────────────────────────────────────────────────────────
    "Finance_and_Financial_Economics": [
        "Journal of Finance",
        "Journal of Financial Economics",
        "Review of Financial Studies",
        "Management Science",
        "Journal of Corporate Finance",
        "European Financial Management",
        "Journal of Empirical Finance",
        "International Review of Financial Analysis",
        "Review of Quantitative Finance and Accounting",
        "Quantitative Finance",
        "Research in International Business and Finance",
        "Energy Economics",
        "Emerging Markets Review",
        "Emerging Markets Finance and Trade",
        "Economic Modelling",
        "Journal of Financial Markets, Institutions & Money",
        "International Review of Economics & Finance",
        "Foundations and Trends in Finance",
        "Journal of Econometrics",
        "Journal of Economic Dynamics and Control",
        "Journal of Economic Behavior & Organization",
        "Journal of Business Finance and Accounting",
        "European Journal of Finance",
        "Borsa Istanbul Review",
        "Financial Innovation",
        "Journal of Finance and Data Science",
        "HSE Economic Journal",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC 1400/1401 — General / Miscellaneous Business (optional bucket)
    # ─────────────────────────────────────────────────────────────
    "General_and_Misc": [
        "Journal of Business Research",
        "Humanities and Social Sciences Communications",
        "European Research on Management and Business Economics",
    ],

    # ─────────────────────────────────────────────────────────────
    # ASJC-aligned Ops/OR/Supply Chain (mix of 1408 + 1800/2200 crossover)
    # ─────────────────────────────────────────────────────────────
    "Supply_Chain_and_Operations": [
        "Journal of Operations Management",
        "Manufacturing & Service Operations Management",
        "Operations Research",
        "Operations Research Perspectives",
        "Operations Research Forum",
        "European Journal of Operational Research",
        "Annals of Operations Research",
        "OR Spectrum",
        "Production Planning & Control",
        "International Journal of Production Research",
        "International Journal of Production Economics",
        "International Journal of Operations & Production Management",
        "Supply Chain Management: An International Journal",
        "International Journal of Physical Distribution & Logistics Management",
        "International Journal of Logistics Management",
        "Journal of Purchasing & Supply Management",
        "Transportation Research Part E: Logistics and Transportation Review",
        "EURO Journal on Transportation and Logistics",
        "Computers & Operations Research",
        "Cleaner Logistics and Supply Chain",
        "Communications in Transportation Research",
        "International Journal of Logistics Research and Applications",
        "Journal of Air Transport Management",
    ],

    # ─────────────────────────────────────────────────────────────
    # Ethics/CSR & Governance (mostly sits in 1408 + 3312 ethics crossover)
    # ─────────────────────────────────────────────────────────────
    "Ethics_CSR_and_Governance": [
        "Journal of Business Ethics",
        "Corporate Social Responsibility and Environmental Management",
        "Journal of Management and Governance",
        "Digital Policy, Regulation and Governance",
        "International Journal of Disclosure and Governance",
        "Science and Engineering Ethics",
        "Business Strategy and the Environment",
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