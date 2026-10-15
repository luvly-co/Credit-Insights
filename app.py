"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Run locally:
    streamlit run app.py
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
from typing import TypedDict, List, Any, Dict, Optional  # type hints and structured data types
import pandas as pd            # Data manipulation and analysis
import sqlite3                 # Connect to and query the SQLite database
import sqlparse                # Parse and format SQL queries
import json                    # Read and write JSON data
import re                      # Regular expression-based text processing
import os                      # Interact with the operating system and environment variables
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from langchain_openai import ChatOpenAI                              # OpenAI chat models through LangChain
from langchain_core.prompts import ChatPromptTemplate                # Reusable structured prompts
from langchain_core.messages import HumanMessage, SystemMessage      # System and user messages for LLM calls

import warnings                # Manage Python warning messages
warnings.filterwarnings("ignore")  # Suppress warning messages to keep the UI clean


# =====================================================================
# 2. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# =====================================================================
# 3. CONFIGURATION (paths + credentials)
# =====================================================================
APP_DIR = Path(__file__).resolve().parent

DB_PATH = os.environ.get("DB_PATH", str(APP_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = os.environ.get("TEST_QUERIES_PATH", str(APP_DIR / "test_queries.csv"))
CONFIG_JSON_PATH = os.environ.get("CONFIG_PATH", str(APP_DIR / "config.json"))
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(APP_DIR / "audit_log.jsonl"))

PRIMARY_MODEL = "gpt-4o-mini"     # routing, SQL generation, retry, response
EVALUATOR_MODEL = "gpt-4o"        # LLM relevance check in the validation gate


def get_setting(key: str) -> Optional[str]:
    """
    Resolve a credential in this order:
    1) environment variable, 2) Streamlit secrets, 3) local config.json
    """
    val = os.environ.get(key)
    if val:
        return val
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    try:
        if os.path.exists(CONFIG_JSON_PATH):
            with open(CONFIG_JSON_PATH, "r") as f:
                return json.load(f).get(key)
    except Exception:
        pass
    return None


# Store API credentials in environment variables (as in the notebook)
_api_key = get_setting("OPENAI_API_KEY")
_api_base = get_setting("OPENAI_API_BASE")
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key
if _api_base:
    os.environ["OPENAI_API_BASE"] = _api_base


# =====================================================================
# 4. DATABASE SCHEMA (provided to the LLM in prompts)
# =====================================================================
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =====================================================================
# 5. VERIFIED QUERY TEMPLATE LIBRARY
# =====================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT sm.sector_name,
       SUM(lm.total_outstanding)/ 1000000.0 AS total_outstanding_millions,
       SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) / 1000000.0 AS npa_exposure_millions
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_millions DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
  loan_category,
  SUM(total_outstanding)/1000000.0 AS total_outstanding_millions,
  COUNT(loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
  ifrs9_stage,
  COUNT(loan_account_number) AS loan_count,
  SUM(ead_amount)/1000000.0 AS total_ead_millions,
  SUM(ecl_amount)/1000000.0 AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
  sm.sector_name,
  AVG(p.provision_coverage_ratio) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding/1000000.0 AS outstanding_millions,
  lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
  group_name,
  COUNT(loan_account_number) AS loan_count,
  SUM(total_outstanding)/1000000.0 AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != '' AND group_name != 'None'
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding/1000000.0 AS total_outstanding_millions,
  lm.days_past_due,
  lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_millions
FROM loan_master
GROUP BY
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END
ORDER BY
    MIN(days_past_due) ASC
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
  borrower_id,
  previous_rating,
  internal_rating AS current_internal_rating,
  pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
  reporting_date,
  SUM(ecl_amount)/1000000.0 AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# =====================================================================
# 6. TOOL DEFINITIONS
#    (llm / evaluator_llm / are initialised in section 8 below)
# =====================================================================

# ---------------------------------------------------------------------
# Intent Classification Tool
# ---------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
System: You are an expert router for a financial credit risk analytics engine. Your job is to classify the user's question and decide if it can be directly answered by one of the pre-approved verified query templates, or if it requires generating a new query.

Here are the pre-approved templates:
{library_descriptions}

Analyze the user's question:
"{user_question}"

Rules:
- Choose "verified" and return "VQ1" if the user asks about outstanding portfolio exposure, book value, or NPA levels in the real estate sector, as VQ1 provides a comprehensive sector-wise breakdown containing this exact information.
- Choose "verified" and return the corresponding "query_id" if the question matches the description of any pre-approved template.
- Choose "generated" and set "query_id" to null only if the question requires a highly custom filter, calculation, or table join that is completely absent from the templates.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---------------------------------------------------------------------
# Query Generation Tool
# ---------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
System: You are an expert SQL Assistant. Write a SQLite-compatible read-only SQL query to answer the user's question.

Ensure that:
- You only use tables and columns defined in the schema.
- Use correct table aliases and valid join keys.
- To calculate NPA exposure, filter or aggregate using the rule: `asset_classification IN ('Substandard', 'Doubtful', 'Loss')`.
- Convert currency values to millions of dollars where appropriate.
- Ensure the output query is valid SQL syntax and starts with SELECT or WITH.

User Question:
{user_question}

Database Schema:
{schema_context}

Output only the raw SQL query. Do not wrap in markdown or include conversational text.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ---------------------------------------------------------------------
# Query Validation Tool
# ---------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
System: You are a credit risk validator. Analyze if the SQL matches the intent of the user's question.
Context: {track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

Output only a JSON dictionary with keys "verdict" ("yes" or "no"), "confidence" (between 0.0 and 1.0), and "reason" (short explanation).
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ---------------------------------------------------------------------
# Retry Generation Tool
# ---------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
System: You are an expert database debugger. Rewrite the SQL query so that it successfully parses, strictly conforms to the schema context, and contains only a single read-only SELECT statement.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Return ONLY the revised SQLite SQL query with no explanation and no markdown tags.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---------------------------------------------------------------------
# Query Execution Tool
# ---------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ---------------------------------------------------------------------
# Response Generation Tool
# ---------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
System: You are a professional credit risk analyst. Present a concise and professional summary answering the user's question using the provided dataframe data.

Ensure that:
- You only mention facts and figures present in the dataframe.
- Be brief, direct, and use tables where helpful.

User Question:
{user_question}

Query Results:
{dataframe.to_string()}
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =====================================================================
# 7. PIPELINE ORCHESTRATION
# =====================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in 'trace'
                      (the notebook printed these; the app displays them).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, log and trace.
    '''

    trace: List[str] = []

    def _say(msg: str):
        if verbose:
            trace.append(msg)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    _say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    _say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    _say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    _say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        _say(f"    Failed check: {gate.get('failed_check')}")
        _say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        _say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        _say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            _say(f"    Retry failed check: {gate.get('failed_check')}")
            _say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        _say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'warnings': [], 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    _say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        _say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    _say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'warnings': exec_result['warnings'], 'trace': trace, **log}


# =====================================================================
# 8. RESOURCE INITIALISATION (LLMs, read-only DB, ground truth)
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_llms():
    _llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=0)
    _evaluator = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    return _llm, _evaluator


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    """Open the SQLite database in READ-ONLY mode (same as the notebook)."""
    uri_path = quote(Path(db_path).resolve().as_posix(), safe="/:")
    # check_same_thread=False because Streamlit runs scripts in worker threads
    return sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def load_ground_truth(csv_path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def append_audit_record(record: Dict[str, Any]):
    """Keep an audit trail: session history + append-only JSONL file."""
    st.session_state.setdefault("audit_log", []).append(record)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # read-only / ephemeral filesystems: session log still available


def build_audit_record(res: Dict[str, Any]) -> Dict[str, Any]:
    gate = res.get("gate_result") or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_question": res.get("user_question"),
        "route": res.get("route"),
        "query_id": res.get("query_id"),
        "match_reason": res.get("match_reason"),
        "candidate_sql": res.get("candidate_sql"),
        "executed_sql": res.get("executed_sql"),
        "gate_passed": gate.get("passed"),
        "failed_check": gate.get("failed_check"),
        "gate_details": gate.get("details"),
        "retry_used": res.get("retry_used"),
        "escalated": res.get("escalated"),
        "row_count": res.get("row_count"),
        "confidence": res.get("confidence"),
        "narrative": res.get("narrative"),
    }


def fmt_confidence(conf) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.0%}"
    return str(conf) if conf is not None else "N/A"


# ---- Fail early (with clear messages) if configuration is incomplete ----
_missing = []
if not os.environ.get("OPENAI_API_KEY"):
    _missing.append("`OPENAI_API_KEY` (env var, Streamlit secret, or config.json)")
if not os.path.exists(DB_PATH):
    _missing.append(f"database file not found at `{DB_PATH}` (set `DB_PATH` or place `credit_risk_portfolio.db` next to app.py)")

if _missing:
    st.title("🏦 Credit Risk Query Engine")
    st.error("Configuration incomplete:\n\n- " + "\n- ".join(_missing))
    st.stop()

llm, evaluator_llm = get_llms()
conn = get_connection(DB_PATH)
ground_truth = load_ground_truth(TEST_QUERIES_PATH)


# =====================================================================
# 9. STREAMLIT UI
# =====================================================================
st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. "
    "Recurring questions use pre-approved, verified SQL templates; novel questions get "
    "read-only generated SQL that must pass a 5-check validation gate. "
    "Every answer shows the SQL used, the raw data and a confidence score."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("System status")
    st.success("Database connected (read-only mode)")
    try:
        tables_df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )
        st.write("**Tables:** " + ", ".join(tables_df["name"].tolist()))
    except Exception as e:
        st.error(f"Could not read tables: {e}")
    st.write(f"**Router / SQL / Response model:** `{PRIMARY_MODEL}`")
    st.write(f"**Validator model:** `{EVALUATOR_MODEL}`")
    st.write(f"**Verified templates:** {len(verified_query_library)}")

    st.divider()
    st.subheader("Example questions")
    examples = []
    if ground_truth is not None and "User Query" in ground_truth.columns:
        examples = ground_truth["User Query"].dropna().tolist()
    else:
        examples = [
            "Show me how our overdue book is distributed across different Days Past Due buckets",
            "What is the average interest rate by sector?",
        ]

    def _set_question(q):
        st.session_state["question_input"] = q

    for i, q in enumerate(examples):
        st.button(q if len(q) < 80 else q[:77] + "...", key=f"ex_{i}",
                  on_click=_set_question, args=(q,), use_container_width=True)

    st.divider()
    if st.button("Clear session history", use_container_width=True):
        st.session_state["audit_log"] = []
        st.session_state.pop("last_result", None)
        st.rerun()

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["💬 Ask a question", "📚 Verified query library", "🧪 Evaluation", "🧾 Audit trail"]
)

# ---------------- Tab 1: Ask ----------------
with tab_ask:
    st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g., What is our total outstanding exposure and NPA level in the Real Estate sector?",
    )
    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        question = st.session_state.get("question_input", "").strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Running pipeline: classify → build SQL → validate → execute → respond..."):
                    result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
                st.session_state["last_result"] = result
                append_audit_record(build_audit_record(result))
            except Exception as e:
                st.session_state.pop("last_result", None)
                st.error(f"Pipeline error: {e}")

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Answer")

        if res["escalated"]:
            st.error("⚠️ Escalated to a human analyst — the query could not be reliably resolved.")
            st.write(res["narrative"])
        else:
            st.markdown(res["narrative"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", fmt_confidence(res["confidence"]))
        c2.metric("Route", (res["route"] or "").title())
        c3.metric("Query ID", res["query_id"] or "—")
        c4.metric("Rows returned", res["row_count"] if res["row_count"] is not None else "—")

        if res.get("retry_used"):
            st.info("A retry was used: the first generated SQL failed validation and was regenerated once.")
        for w in res.get("warnings", []):
            st.warning(w)

        st.subheader("SQL used")
        sql_to_show = res["executed_sql"] or res["candidate_sql"]
        if res["escalated"]:
            st.caption("Candidate SQL (failed validation — NOT executed)")
        st.code(sqlparse.format(sql_to_show, reindent=True, keyword_case="upper") if sql_to_show else "", language="sql")

        st.subheader("Raw data returned")
        if res["dataframe"] is not None:
            st.dataframe(res["dataframe"], use_container_width=True)
            st.download_button(
                "Download result as CSV",
                data=res["dataframe"].to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )
        else:
            st.info("No data returned (query was escalated).")

        with st.expander("Pipeline trace & validation details"):
            st.text("\n".join(res.get("trace", [])))
            gate = res.get("gate_result") or {}
            st.json({
                "passed": gate.get("passed"),
                "failed_check": gate.get("failed_check"),
                "details": gate.get("details"),
                "relevance_confidence": gate.get("relevance_confidence"),
                "match_reason": res.get("match_reason"),
            })

# ---------------- Tab 2: Verified query library ----------------
with tab_library:
    st.subheader("Verified Query Template Library")
    st.caption("Pre-approved SQL templates for recurring questions. They run without modification.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid} — {entry['description']}"):
            st.code(entry["sql"].strip(), language="sql")

# ---------------- Tab 3: Evaluation ----------------
with tab_eval:
    st.subheader("Evaluation Against Ground Truth")
    if ground_truth is None:
        st.warning(f"`test_queries.csv` not found at `{TEST_QUERIES_PATH}`. "
                   "Add it next to app.py (or set `TEST_QUERIES_PATH`) to enable evaluation.")
    else:
        st.dataframe(ground_truth, use_container_width=True)
        st.caption("Runs every test case end-to-end through the pipeline (makes several LLM calls per case).")
        if st.button("Run evaluation on all test cases"):
            evaluation_rows = []
            test_outputs = []
            progress = st.progress(0.0, text="Starting...")
            n = len(ground_truth)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                progress.progress(i / n, text=f"Running test case {i + 1}/{n}: {gt['Test Case']}")
                try:
                    tr = run_pipeline(gt['User Query'], conn, verified_query_library, database_schema, verbose=False)
                except Exception as e:
                    tr = {'route': None, 'query_id': None, 'confidence': None, 'row_count': None,
                          'narrative': f"Pipeline error: {e}", 'dataframe': None}
                test_outputs.append(tr)
                if tr.get('log'):
                    append_audit_record(build_audit_record(tr))
                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })
            progress.progress(1.0, text="Done")

            evaluation_df = pd.DataFrame(evaluation_rows)
            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100
            # 'ESCALATED' confidences are non-numeric -> coerce so the mean still works
            average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence Score", f"{average_confidence:.2f}")
            st.dataframe(evaluation_df.astype({'Confidence': 'object'}), use_container_width=True)

            st.subheader("Per-test-case details")
            for (_, gt), tr in zip(ground_truth.iterrows(), test_outputs):
                with st.expander(f"{gt['Test Case']} — {gt['User Query']}"):
                    st.write(f"**Confidence:** {fmt_confidence(tr.get('confidence'))}")
                    st.markdown(tr.get('narrative') or "")
                    if tr.get('executed_sql'):
                        st.code(tr['executed_sql'], language="sql")
                    if tr.get('dataframe') is not None:
                        st.dataframe(tr['dataframe'], use_container_width=True)

# ---------------- Tab 4: Audit trail ----------------
with tab_audit:
    st.subheader("Audit trail (this session)")
    st.caption(f"Every output is also appended to `{AUDIT_LOG_PATH}` when the filesystem is writable.")
    audit = st.session_state.get("audit_log", [])
    if audit:
        audit_df = pd.DataFrame(audit)
        st.dataframe(audit_df, use_container_width=True)
        st.download_button(
            "Download audit log (CSV)",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No queries have been run in this session yet.")
