import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List


def create_dormancy_breakdown_chart(dormant_data: Dict[str, Any]) -> go.Figure:
    """Create dormancy breakdown bar chart"""
    if not dormant_data or 'summary_kpis' not in dormant_data:
        return go.Figure()

    kpis = dormant_data['summary_kpis']

    breakdown_data = {
        'Category': ['Safe Deposit', 'Investment', 'Fixed Deposit', 'Demand Deposit', 'Unclaimed Instruments'],
        'Count': [
            kpis.get('count_sdb_dormant', 0),
            kpis.get('count_investment_dormant', 0),
            kpis.get('count_fixed_deposit_dormant', 0),
            kpis.get('count_demand_deposit_dormant', 0),
            kpis.get('count_unclaimed_instruments', 0)
        ]
    }

    fig = px.bar(
        x=breakdown_data['Category'],
        y=breakdown_data['Count'],
        title="Dormant Accounts by Category",
        labels={'x': 'Account Category', 'y': 'Number of Accounts'},
        color=breakdown_data['Count'],
        color_continuous_scale='Reds'
    )

    return fig


def create_compliance_scatter_chart(compliance_data: Dict[str, Any]) -> go.Figure:
    """Create compliance issues scatter chart"""
    if not compliance_data:
        return go.Figure()

    issues_data = {
        'Issue Type': ['CB Transfer', 'Incomplete Contact', 'Unflagged Dormant', 'Statement Freeze', 'Internal Ledger'],
        'Count': [
            compliance_data.get('transfer_candidates_cb', {}).get('count', 0),
            compliance_data.get('incomplete_contact', {}).get('count', 0),
            compliance_data.get('flag_candidates', {}).get('count', 0),
            compliance_data.get('statement_freeze_needed', {}).get('count', 0),
            compliance_data.get('ledger_candidates_internal', {}).get('count', 0)
        ],
        'Priority': ['Critical', 'High', 'Medium', 'Medium', 'Low']
    }

    fig = px.scatter(
        x=issues_data['Issue Type'],
        y=issues_data['Count'],
        size=issues_data['Count'],
        color=issues_data['Priority'],
        title="Compliance Issues by Type and Priority",
        size_max=50
    )

    return fig


def create_risk_gauge(confidence_score: float) -> go.Figure:
    """Create risk assessment gauge"""
    risk_score = 1 - confidence_score

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Risk Level"},
        delta={'reference': 0.3},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.6], 'color': "yellow"},
                {'range': [0.6, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))

    return fig


def create_notification_pie_chart(notifications: List[str]) -> go.Figure:
    """Create notification distribution pie chart"""
    if not notifications:
        return go.Figure()

    notification_types = {}
    for notif in notifications:
        if "URGENT" in notif:
            notification_types["Critical"] = notification_types.get("Critical", 0) + 1
        elif "WARNING" in notif:
            notification_types["Warning"] = notification_types.get("Warning", 0) + 1
        else:
            notification_types["Info"] = notification_types.get("Info", 0) + 1

    fig = px.pie(
        values=list(notification_types.values()),
        names=list(notification_types.keys()),
        title="Notification Distribution"
    )

    return fig