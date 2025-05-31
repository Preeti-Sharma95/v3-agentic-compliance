# agents/dormant_analyzers.py
"""
Modular dormant account analyzers based on CBUAE regulations
Each analyzer is a separate agent focused on specific dormancy criteria
"""

import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


# CBUAE Dormancy Configuration
class DormancyConfig:
    STANDARD_INACTIVITY_YEARS = 3
    PAYMENT_INSTRUMENT_UNCLAIMED_YEARS = 1
    SDB_UNPAID_FEES_YEARS = 3
    ELIGIBILITY_FOR_CB_TRANSFER_YEARS = 5
    HIGH_VALUE_THRESHOLD = 25000
    WARNING_PERIOD_YEARS = 2.5


class BaseDormancyAnalyzer(ABC):
    """Base class for all dormancy analyzers"""

    def __init__(self, memory_agent=None):
        self.memory_agent = memory_agent
        self.config = DormancyConfig()

    @abstractmethod
    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        """
        Analyze dormancy criteria
        Returns: (flagged_accounts_df, count, description, details)
        """
        pass

    def log_analysis(self, session_id: str, analysis_type: str, result: Dict[str, Any]):
        """Log analysis results to memory"""
        if self.memory_agent:
            self.memory_agent.log(session_id, f'dormant_analysis_{analysis_type}', result)


class SafeDepositDormancyAnalyzer(BaseDormancyAnalyzer):
    """Article 2.6 CBUAE - Safe Deposit Box Dormancy"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = [
                'Account_ID', 'Account_Type', 'SDB_Charges_Outstanding',
                'Date_SDB_Charges_Became_Outstanding', 'SDB_Tenant_Communication_Received'
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped SDB Dormancy: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_SDB_Charges_Became_Outstanding']):
                df_work['Date_SDB_Charges_Became_Outstanding'] = pd.to_datetime(
                    df_work['Date_SDB_Charges_Became_Outstanding'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.SDB_UNPAID_FEES_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
                (pd.to_numeric(df_work['SDB_Charges_Outstanding'], errors='coerce').fillna(0) > 0) &
                (df_work['Date_SDB_Charges_Became_Outstanding'].notna()) &
                (df_work['Date_SDB_Charges_Became_Outstanding'] < threshold_date) &
                (df_work['SDB_Tenant_Communication_Received'].astype(str).str.lower().isin(
                    ['no', 'false', '0', 'nan', '']))
                ].copy()

            count = len(flagged_accounts)
            description = f"Safe Deposit Boxes meeting dormancy criteria (Art 2.6: >{self.config.SDB_UNPAID_FEES_YEARS}yr unpaid, no tenant reply): {count} boxes"

            details = {
                "regulation": "Article 2.6 CBUAE",
                "criteria": f"Unpaid charges for {self.config.SDB_UNPAID_FEES_YEARS}+ years with no tenant communication",
                "average_outstanding": pd.to_numeric(flagged_accounts['SDB_Charges_Outstanding'],
                                                     errors='coerce').mean() if count else 0,
                "total_outstanding": pd.to_numeric(flagged_accounts['SDB_Charges_Outstanding'],
                                                   errors='coerce').sum() if count else 0,
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else []
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"SDB Dormancy Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Safe Deposit Dormancy check: {e})", {}


class InvestmentInactivityAnalyzer(BaseDormancyAnalyzer):
    """Article 2.3 CBUAE - Investment Account Inactivity"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped Investment Inactivity: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.STANDARD_INACTIVITY_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < threshold_date)
                ].copy()

            count = len(flagged_accounts)
            description = f"Investment accounts dormant (Art 2.3: >{self.config.STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"

            details = {
                "regulation": "Article 2.3 CBUAE",
                "criteria": f"No customer-initiated activity for {self.config.STANDARD_INACTIVITY_YEARS}+ years",
                "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else []
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Investment Inactivity Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Investment Inactivity check: {e})", {}


class FixedDepositInactivityAnalyzer(BaseDormancyAnalyzer):
    """Article 2.2 CBUAE - Fixed/Term Deposit Inactivity"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped Fixed Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.STANDARD_INACTIVITY_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)) &
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < threshold_date)
                ].copy()

            count = len(flagged_accounts)
            description = f"Fixed Deposit accounts dormant (Art 2.2: >{self.config.STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"

            details = {
                "regulation": "Article 2.2 CBUAE",
                "criteria": f"No customer-initiated activity for {self.config.STANDARD_INACTIVITY_YEARS}+ years",
                "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else []
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Fixed Deposit Inactivity Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Fixed Deposit Inactivity check: {e})", {}


class DemandDepositInactivityAnalyzer(BaseDormancyAnalyzer):
    """Article 2.1.1 CBUAE - Demand Deposit Inactivity"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped Demand Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.STANDARD_INACTIVITY_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)) &
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < threshold_date)
                ].copy()

            count = len(flagged_accounts)
            description = f"Demand Deposit accounts dormant (Art 2.1.1: >{self.config.STANDARD_INACTIVITY_YEARS}yr inactive): {count} accounts"

            details = {
                "regulation": "Article 2.1.1 CBUAE",
                "criteria": f"No customer-initiated activity for {self.config.STANDARD_INACTIVITY_YEARS}+ years",
                "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else []
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Demand Deposit Inactivity Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Demand Deposit Inactivity check: {e})", {}


class UnclaimedPaymentInstrumentsAnalyzer(BaseDormancyAnalyzer):
    """Article 2.4 CBUAE - Unclaimed Payment Instruments"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped Unclaimed Instruments: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.PAYMENT_INSTRUMENT_UNCLAIMED_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order", case=False,
                                                                  na=False)) &
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < threshold_date)
                ].copy()

            count = len(flagged_accounts)
            description = f"Unclaimed payment instruments (Art 2.4: >{self.config.PAYMENT_INSTRUMENT_UNCLAIMED_YEARS}yr unclaimed): {count} items"

            details = {
                "regulation": "Article 2.4 CBUAE",
                "criteria": f"Unclaimed for {self.config.PAYMENT_INSTRUMENT_UNCLAIMED_YEARS}+ years",
                "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else []
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Unclaimed Instruments Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Unclaimed Instruments check: {e})", {}


class CBUAETransferEligibilityAnalyzer(BaseDormancyAnalyzer):
    """Article 8 CBUAE - Eligibility for Central Bank Transfer"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped CB Transfer: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            threshold_date = report_date - timedelta(days=self.config.ELIGIBILITY_FOR_CB_TRANSFER_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < threshold_date)
                ].copy()

            count = len(flagged_accounts)
            description = f"Accounts eligible for CBUAE transfer (Art 8: >{self.config.ELIGIBILITY_FOR_CB_TRANSFER_YEARS}yr dormant): {count} accounts"

            # Calculate total balance if available
            total_balance = 0
            if 'Current_Balance' in flagged_accounts.columns:
                total_balance = pd.to_numeric(flagged_accounts['Current_Balance'], errors='coerce').sum()

            details = {
                "regulation": "Article 8 CBUAE",
                "criteria": f"Dormant for {self.config.ELIGIBILITY_FOR_CB_TRANSFER_YEARS}+ years",
                "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                "total_balance_to_transfer": total_balance,
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else [],
                "urgency": "HIGH" if count > 0 else "NONE"
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"CBUAE Transfer Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in CB Transfer check: {e})", {}


class Article3ProcessAnalyzer(BaseDormancyAnalyzer):
    """Article 3 Process Tracking"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            # Accounts that are dormant but haven't gone through Article 3 process
            flagged_accounts = df[
                df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1']) &
                (~df.get('Article_3_Process_Completed', '').astype(str).str.lower().isin(['yes', 'true', '1']))
                ].copy()

            count = len(flagged_accounts)
            description = f"Accounts needing Article 3 process: {count} accounts"

            details = {
                "regulation": "Article 3 CBUAE",
                "criteria": "Dormant accounts requiring Article 3 compliance process",
                "process_steps": ["Customer notification", "Account review", "Documentation", "Approval"],
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else [],
                "priority": "MEDIUM"
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Article 3 Process Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Art3 Process check: {e})", {}


class ProactiveContactAnalyzer(BaseDormancyAnalyzer):
    """Proactive Customer Contact Analysis"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Date_Last_Cust_Initiated_Activity']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped Contact Attempts: Missing {', '.join(missing_cols)})", {}

            df_work = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_work['Date_Last_Cust_Initiated_Activity']):
                df_work['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df_work['Date_Last_Cust_Initiated_Activity'], errors='coerce')

            # Warning period - 2.5 years inactive (before full dormancy)
            warning_threshold = report_date - timedelta(days=int(self.config.WARNING_PERIOD_YEARS * 365))
            full_threshold = report_date - timedelta(days=self.config.STANDARD_INACTIVITY_YEARS * 365)

            flagged_accounts = df_work[
                (df_work['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df_work['Date_Last_Cust_Initiated_Activity'] < warning_threshold) &
                (df_work['Date_Last_Cust_Initiated_Activity'] >= full_threshold) &
                (~df_work.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])) &
                (~df_work.get('Proactive_Contact_Attempted', '').astype(str).str.lower().isin(['yes', 'true', '1']))
                ].copy()

            count = len(flagged_accounts)
            description = f"Accounts needing proactive contact: {count} accounts"

            details = {
                "regulation": "Proactive Customer Engagement",
                "criteria": f"Inactive for {self.config.WARNING_PERIOD_YEARS}+ years but not yet dormant",
                "warning_threshold": warning_threshold.strftime("%Y-%m-%d"),
                "contact_methods": ["Email", "SMS", "Phone", "Physical Mail"],
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else [],
                "prevention_opportunity": True
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"Proactive Contact Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in Contact Attempts check: {e})", {}


class HighValueDormantAnalyzer(BaseDormancyAnalyzer):
    """High-Value Dormant Account Monitoring"""

    def analyze(self, df: pd.DataFrame, report_date: datetime) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            required_columns = ['Account_ID', 'Current_Balance', 'Expected_Account_Dormant']

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return pd.DataFrame(), 0, f"(Skipped High Value: Missing {', '.join(missing_cols)})", {}

            flagged_accounts = df[
                (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
                (pd.to_numeric(df['Current_Balance'], errors='coerce').fillna(0) >= self.config.HIGH_VALUE_THRESHOLD)
                ].copy()

            count = len(flagged_accounts)
            total_balance = pd.to_numeric(flagged_accounts['Current_Balance'], errors='coerce').sum() if count else 0
            avg_balance = pd.to_numeric(flagged_accounts['Current_Balance'], errors='coerce').mean() if count else 0

            description = f"High-value dormant accounts (>= {self.config.HIGH_VALUE_THRESHOLD:,}): {count} accounts"

            details = {
                "regulation": "High-Value Account Monitoring",
                "criteria": f"Dormant accounts with balance >= {self.config.HIGH_VALUE_THRESHOLD:,}",
                "total_balance": total_balance,
                "average_balance": avg_balance,
                "threshold": self.config.HIGH_VALUE_THRESHOLD,
                "sample_accounts": flagged_accounts['Account_ID'].head(3).tolist() if count else [],
                "risk_level": "HIGH" if total_balance > 1000000 else "MEDIUM",
                "enhanced_monitoring_required": True
            }

            return flagged_accounts, count, description, details

        except Exception as e:
            logger.error(f"High Value Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in High Value check: {e})", {}


class DormantToActiveTransitionAnalyzer(BaseDormancyAnalyzer):
    """Dormant-to-Active Transition Tracking"""

    def analyze(self, df: pd.DataFrame, report_date: datetime,
                dormant_history_df: pd.DataFrame = None,
                activity_lookback_days: int = 30) -> Tuple[pd.DataFrame, int, str, Dict[str, Any]]:
        try:
            if dormant_history_df is None or dormant_history_df.empty:
                return pd.DataFrame(), 0, "(No dormant history available for transition analysis)", {}

            # Look for accounts that were dormant but now show recent activity
            recent_activity_threshold = report_date - timedelta(days=activity_lookback_days)

            # This is a simplified implementation - in practice, you'd need historical dormant flags
            potentially_reactivated = df[
                (df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['no', 'false', '0'])) &
                (pd.to_datetime(df.get('Date_Last_Cust_Initiated_Activity', ''),
                                errors='coerce') > recent_activity_threshold)
                ].copy()

            # Cross-reference with historical dormant data
            # This would require more sophisticated logic with actual historical data

            count = 0  # Placeholder - would be calculated based on actual historical comparison
            description = f"Dormant-to-active transitions: {count} accounts"

            details = {
                "regulation": "Reactivation Monitoring",
                "criteria": f"Previously dormant accounts with activity in last {activity_lookback_days} days",
                "lookback_period": activity_lookback_days,
                "recent_activity_threshold": recent_activity_threshold.strftime("%Y-%m-%d"),
                "sample_accounts": [],
                "requires_review": count > 0,
                "compliance_implications": "Review dormancy status and update flags"
            }

            return pd.DataFrame(), count, description, details

        except Exception as e:
            logger.error(f"Transition Analysis Error: {e}")
            return pd.DataFrame(), 0, f"(Error in transitions check: {e})", {}