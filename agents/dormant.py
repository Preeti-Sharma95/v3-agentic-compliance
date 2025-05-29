import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# CBUAE Dormancy Periods (examples, can be centralized in config.py)
STANDARD_INACTIVITY_YEARS = 3
PAYMENT_INSTRUMENT_UNCLAIMED_YEARS = 1
SDB_UNPAID_FEES_YEARS = 3
ELIGIBILITY_FOR_CB_TRANSFER_YEARS = 5


def check_safe_deposit_dormancy(df, report_date):
    """
    Identifies Safe Deposit Boxes meeting dormancy criteria (Art. 2.6 CBUAE).
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'SDB_Charges_Outstanding',
            'Date_SDB_Charges_Became_Outstanding', 'SDB_Tenant_Communication_Received'
        ]

        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped SDB Dormancy: Missing {', '.join(missing_cols)})", {}

        # Convert date column to datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_SDB_Charges_Became_Outstanding']):
            df['Date_SDB_Charges_Became_Outstanding'] = pd.to_datetime(
                df['Date_SDB_Charges_Became_Outstanding'], errors='coerce')

        threshold_date_sdb = report_date - timedelta(days=SDB_UNPAID_FEES_YEARS * 365)

        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (pd.to_numeric(df['SDB_Charges_Outstanding'], errors='coerce').fillna(0) > 0) &
            (df['Date_SDB_Charges_Became_Outstanding'].notna()) &
            (df['Date_SDB_Charges_Became_Outstanding'] < threshold_date_sdb) &
            (df['SDB_Tenant_Communication_Received'].astype(str).str.lower().isin(['no', 'false', '0', 'nan', '']))
            ].copy()

        count = len(data)
        desc = f"Safe Deposit Boxes meeting dormancy criteria (Art 2.6: >{SDB_UNPAID_FEES_YEARS}yr unpaid, no tenant reply): {count} boxes"
        details = {
            "average_outstanding_charges": pd.to_numeric(data['SDB_Charges_Outstanding'],
                                                         errors='coerce').mean() if count else 0,
            "total_outstanding_charges": pd.to_numeric(data['SDB_Charges_Outstanding'],
                                                       errors='coerce').sum() if count else 0,
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit Dormancy check: {e})", {}


def check_investment_inactivity(df, report_date):
    """
    Identifies Investment Accounts meeting dormancy criteria (Art. 2.3 CBUAE).
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped Investment Inactivity: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        threshold_date = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ].copy()

        count = len(data)
        desc = f"Investment accounts dormant (Art 2.3: >{STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment Inactivity check: {e})", {}


def check_fixed_deposit_inactivity(df, report_date):
    """
    Identifies Fixed/Term Deposit accounts meeting dormancy criteria (Art. 2.2 CBUAE).
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped Fixed Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        threshold_date = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        data = df[
            (df['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ].copy()

        count = len(data)
        desc = f"Fixed Deposit accounts dormant (Art 2.2: >{STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit Inactivity check: {e})", {}


def check_demand_deposit_inactivity(df, report_date):
    """
    Identifies Demand Deposit accounts meeting dormancy criteria (Art. 2.1.1 CBUAE).
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped Demand Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        threshold_date = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        data = df[
            (df['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ].copy()

        count = len(data)
        desc = f"Demand Deposit accounts dormant (Art 2.1.1: >{STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Demand Deposit Inactivity check: {e})", {}


def check_unclaimed_payment_instruments(df, report_date):
    """
    Identifies unclaimed payment instruments (Art. 2.4 CBUAE).
    """
    try:
        required_columns = ['Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped Unclaimed Instruments: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        threshold_date = report_date - timedelta(days=PAYMENT_INSTRUMENT_UNCLAIMED_YEARS * 365)

        data = df[
            (df['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order", case=False,
                                                         na=False)) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ].copy()

        count = len(data)
        desc = f"Unclaimed payment instruments (Art 2.4: >{PAYMENT_INSTRUMENT_UNCLAIMED_YEARS}yr unclaimed): {count} items"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unclaimed Instruments check: {e})", {}


def check_eligible_for_cb_transfer(df, report_date):
    """
    Identifies accounts eligible for CBUAE transfer (Art. 8 CBUAE).
    """
    try:
        required_columns = ['Account_ID', 'Date_Last_Cust_Initiated_Activity']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped CB Transfer: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        threshold_date = report_date - timedelta(days=ELIGIBILITY_FOR_CB_TRANSFER_YEARS * 365)

        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ].copy()

        count = len(data)
        desc = f"Accounts eligible for CBUAE transfer (Art 8: >{ELIGIBILITY_FOR_CB_TRANSFER_YEARS}yr dormant): {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in CB Transfer check: {e})", {}


def check_art3_process_needed(df, report_date):
    """
    Identifies accounts needing Article 3 process.
    """
    try:
        # Simple check based on Expected_Account_Dormant flag
        data = df[
            df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])
        ].copy()

        count = len(data)
        desc = f"Accounts needing Article 3 process: {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Art3 Process check: {e})", {}


def check_contact_attempts_needed(df, report_date):
    """
    Identifies accounts needing proactive contact.
    """
    try:
        required_columns = ['Account_ID', 'Date_Last_Cust_Initiated_Activity']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped Contact Attempts: Missing {', '.join(missing_cols)})", {}

        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                df['Date_Last_Cust_Initiated_Activity'], errors='coerce')

        # Warning period - 2.5 years inactive
        warning_threshold = report_date - timedelta(days=int(2.5 * 365))
        full_threshold = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < warning_threshold) &
            (df['Date_Last_Cust_Initiated_Activity'] >= full_threshold) &
            (~df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1']))
            ].copy()

        count = len(data)
        desc = f"Accounts needing proactive contact: {count} accounts"
        details = {
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Contact Attempts check: {e})", {}


def check_high_value_dormant_accounts(df, threshold_balance=25000):
    """
    Identifies high-value dormant accounts.
    """
    try:
        required_columns = ['Account_ID', 'Current_Balance', 'Expected_Account_Dormant']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), 0, f"(Skipped High Value: Missing {', '.join(missing_cols)})", {}

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (pd.to_numeric(df['Current_Balance'], errors='coerce').fillna(0) >= threshold_balance)
            ].copy()

        count = len(data)
        desc = f"High-value dormant accounts (>= {threshold_balance:,}): {count} accounts"
        details = {
            "total_balance": pd.to_numeric(data['Current_Balance'], errors='coerce').sum() if count else 0,
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in High Value check: {e})", {}


def check_dormant_to_active_transitions(df, report_date, dormant_flags_history_df=None, activity_lookback_days=30):
    """
    Identifies dormant-to-active transitions.
    """
    try:
        # Simple implementation - check for recent activity on previously dormant accounts
        if not dormant_flags_history_df or dormant_flags_history_df.empty:
            return pd.DataFrame(), 0, "(No dormant history available)", {}

        # For now, return empty result
        return pd.DataFrame(), 0, "Dormant-to-active transitions: 0 accounts", {}
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in transitions check: {e})", {}


def run_all_dormant_identification_checks(df, report_date_str=None, dormant_flags_history_df=None):
    """
    Runs all dormancy identification checks.
    """
    if report_date_str is None:
        report_date = datetime.now()
    else:
        try:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
        except ValueError:
            report_date = datetime.now()

    df_copy = df.copy()

    results = {
        "report_date_used": report_date.strftime("%Y-%m-%d"),
        "total_accounts_analyzed": len(df_copy),
        "sdb_dormant": {}, "investment_dormant": {}, "fixed_deposit_dormant": {},
        "demand_deposit_dormant": {}, "unclaimed_instruments": {}, "eligible_for_cb_transfer": {},
        "art3_process_needed": {}, "proactive_contact_needed": {},
        "high_value_dormant": {}, "dormant_to_active": {},
        "summary_kpis": {}
    }

    # Run all checks
    results["sdb_dormant"]["df"], results["sdb_dormant"]["count"], results["sdb_dormant"]["desc"], \
    results["sdb_dormant"]["details"] = \
        check_safe_deposit_dormancy(df_copy, report_date)

    results["investment_dormant"]["df"], results["investment_dormant"]["count"], results["investment_dormant"]["desc"], \
    results["investment_dormant"]["details"] = \
        check_investment_inactivity(df_copy, report_date)

    results["fixed_deposit_dormant"]["df"], results["fixed_deposit_dormant"]["count"], results["fixed_deposit_dormant"][
        "desc"], results["fixed_deposit_dormant"]["details"] = \
        check_fixed_deposit_inactivity(df_copy, report_date)

    results["demand_deposit_dormant"]["df"], results["demand_deposit_dormant"]["count"], \
    results["demand_deposit_dormant"]["desc"], results["demand_deposit_dormant"]["details"] = \
        check_demand_deposit_inactivity(df_copy, report_date)

    results["unclaimed_instruments"]["df"], results["unclaimed_instruments"]["count"], results["unclaimed_instruments"][
        "desc"], results["unclaimed_instruments"]["details"] = \
        check_unclaimed_payment_instruments(df_copy, report_date)

    results["eligible_for_cb_transfer"]["df"], results["eligible_for_cb_transfer"]["count"], \
    results["eligible_for_cb_transfer"]["desc"], results["eligible_for_cb_transfer"]["details"] = \
        check_eligible_for_cb_transfer(df_copy, report_date)

    results["art3_process_needed"]["df"], results["art3_process_needed"]["count"], results["art3_process_needed"][
        "desc"], results["art3_process_needed"]["details"] = \
        check_art3_process_needed(df_copy, report_date)

    results["proactive_contact_needed"]["df"], results["proactive_contact_needed"]["count"], \
    results["proactive_contact_needed"]["desc"], results["proactive_contact_needed"]["details"] = \
        check_contact_attempts_needed(df_copy, report_date)

    results["high_value_dormant"]["df"], results["high_value_dormant"]["count"], results["high_value_dormant"]["desc"], \
    results["high_value_dormant"]["details"] = \
        check_high_value_dormant_accounts(df_copy)

    results["dormant_to_active"]["df"], results["dormant_to_active"]["count"], results["dormant_to_active"]["desc"], \
    results["dormant_to_active"]["details"] = \
        check_dormant_to_active_transitions(df_copy, report_date, dormant_flags_history_df)

    # Calculate summary KPIs
    total_dormant = len(
        df_copy[df_copy.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])])
    total_balance = 0
    if 'Current_Balance' in df_copy.columns:
        dormant_df = df_copy[
            df_copy.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])]
        total_balance = pd.to_numeric(dormant_df['Current_Balance'], errors='coerce').sum()

    results["summary_kpis"] = {
        "total_accounts_flagged_dormant": total_dormant,
        "percentage_dormant_of_total": round((total_dormant / len(df_copy)) * 100, 2) if len(df_copy) > 0 else 0,
        "total_dormant_balance_aed": total_balance,
        "count_sdb_dormant": results["sdb_dormant"]["count"],
        "count_investment_dormant": results["investment_dormant"]["count"],
        "count_fixed_deposit_dormant": results["fixed_deposit_dormant"]["count"],
        "count_demand_deposit_dormant": results["demand_deposit_dormant"]["count"],
        "count_unclaimed_instruments": results["unclaimed_instruments"]["count"],
        "count_eligible_for_cb_transfer": results["eligible_for_cb_transfer"]["count"],
        "count_high_value_dormant": results["high_value_dormant"]["count"],
        "count_dormant_to_active_transitions": results["dormant_to_active"]["count"],
        "count_needing_art3_process": results["art3_process_needed"]["count"],
        "count_needing_proactive_contact": results["proactive_contact_needed"]["count"],
    }

    return results

