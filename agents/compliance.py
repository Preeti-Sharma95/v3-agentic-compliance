import pandas as pd
from datetime import datetime, timedelta

def detect_incomplete_contact_attempts(df):
    """
    Detects accounts with incomplete contact attempts.
    """
    try:
        # Simple implementation - check if dormant accounts lack contact attempts
        data = df[
            (df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (~df.get('Bank_Contact_Attempted_Post_Dormancy_Trigger', '').astype(str).str.lower().isin(['yes', 'true', '1']))
        ].copy()

        count = len(data)
        desc = f"Accounts with incomplete contact attempts: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in contact attempts check: {e})"

def detect_unflagged_dormant_candidates(df, inactivity_threshold_date):
    """
    Detects accounts that should be flagged as dormant but aren't.
    """
    try:
        if 'Date_Last_Cust_Initiated_Activity' not in df.columns:
            return pd.DataFrame(), 0, "(Missing Date_Last_Cust_Initiated_Activity column)"

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_copy['Date_Last_Cust_Initiated_Activity']):
            df_copy['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df_copy['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        data = df_copy[
            (df_copy['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df_copy['Date_Last_Cust_Initiated_Activity'] < inactivity_threshold_date) &
            (~df_copy.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1']))
        ].copy()

        count = len(data)
        desc = f"Accounts that should be flagged as dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in unflagged candidates check: {e})"

def detect_internal_ledger_candidates(df):
    """
    Detects accounts ready for internal ledger transfer.
    """
    try:
        # Simple check for dormant accounts that might need ledger transfer
        data = df[
            (df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df.get('Expected_Requires_Article_3_Process', '').astype(str).str.lower().isin(['yes', 'true', '1']))
        ].copy()

        count = len(data)
        desc = f"Accounts ready for internal ledger transfer: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in internal ledger check: {e})"

def detect_statement_freeze_candidates(df, freeze_threshold_date):
    """
    Detects accounts needing statement suppression.
    """
    try:
        data = df[
            df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])
        ].copy()

        count = len(data)
        desc = f"Accounts needing statement suppression: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in statement freeze check: {e})"

def detect_cbuae_transfer_candidates(df):
    """
    Detects accounts ready for CBUAE transfer.
    """
    try:
        data = df[
            df.get('Expected_Transfer_to_CB_Due', '').astype(str).str.lower().isin(['yes', 'true', '1'])
        ].copy()

        count = len(data)
        desc = f"Accounts ready for CBUAE transfer: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in CBUAE transfer check: {e})"

def run_all_compliance_checks(df, general_threshold_date, freeze_threshold_date, agent_name="ComplianceSystem"):
    """
    Run all compliance checks.
    """
    results = {
        "total_accounts_processed": len(df),
        "incomplete_contact": {},
        "flag_candidates": {},
        "ledger_candidates_internal": {},
        "statement_freeze_needed": {},
        "transfer_candidates_cb": {}
    }

    df_copy = df.copy()

    # Run compliance checks
    results["incomplete_contact"]["df"], results["incomplete_contact"]["count"], results["incomplete_contact"]["desc"] = \
        detect_incomplete_contact_attempts(df_copy)

    results["flag_candidates"]["df"], results["flag_candidates"]["count"], results["flag_candidates"]["desc"] = \
        detect_unflagged_dormant_candidates(df_copy, general_threshold_date)

    results["ledger_candidates_internal"]["df"], results["ledger_candidates_internal"]["count"], results["ledger_candidates_internal"]["desc"] = \
        detect_internal_ledger_candidates(df_copy)

    results["statement_freeze_needed"]["df"], results["statement_freeze_needed"]["count"], results["statement_freeze_needed"]["desc"] = \
        detect_statement_freeze_candidates(df_copy, freeze_threshold_date)

    results["transfer_candidates_cb"]["df"], results["transfer_candidates_cb"]["count"], results["transfer_candidates_cb"]["desc"] = \
        detect_cbuae_transfer_candidates(df_copy)

    return results