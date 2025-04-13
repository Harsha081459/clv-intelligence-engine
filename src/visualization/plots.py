"""
Visualization Module
=====================
Reusable plot functions for CLV analysis.
All plots use Plotly for interactivity and Seaborn/Matplotlib for static plots.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CURRENCY_SYMBOL

logger = logging.getLogger(__name__)

# ============================================================
# Color Palette
# ============================================================
COLORS = {
    'primary': '#6366F1',       # Indigo
    'secondary': '#8B5CF6',     # Violet
    'success': '#10B981',       # Emerald
    'warning': '#F59E0B',       # Amber
    'danger': '#EF4444',        # Red
    'info': '#3B82F6',          # Blue
    'light': '#F3F4F6',         # Gray-100
    'dark': '#1F2937',          # Gray-800
    'background': '#0F172A',    # Slate-900
    'surface': '#1E293B',       # Slate-800
    'text': '#F8FAFC',          # Slate-50
}

SEGMENT_COLORS = {
    'Champions': '#10B981',
    'At-Risk High-Value': '#EF4444',
    'Promising': '#3B82F6',
    'Hibernating': '#F59E0B',
    'Lost': '#6B7280',
}

PLOTLY_TEMPLATE = 'plotly_dark'


def create_kpi_metric(value, label, prefix="", suffix="", delta=None, delta_suffix=""):
    """Create a KPI metric dictionary for dashboard cards."""
    return {
        'value': value,
        'label': label,
        'prefix': prefix,
        'suffix': suffix,
        'delta': delta,
        'delta_suffix': delta_suffix,
    }


def plot_revenue_concentration(revenue_series: pd.Series, title: str = "Revenue Concentration (Pareto)") -> go.Figure:
    """
    Pareto chart showing what % of customers drive what % of revenue.
    """
    sorted_rev = revenue_series.sort_values(ascending=False)
    cumulative_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100
    customer_pct = np.arange(1, len(sorted_rev) + 1) / len(sorted_rev) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=customer_pct, y=cumulative_pct,
        mode='lines',
        name='Cumulative Revenue %',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)',
    ))
    
    # Add 80/20 reference lines
    fig.add_hline(y=80, line_dash="dash", line_color=COLORS['warning'], 
                  annotation_text="80% of revenue")
    fig.add_vline(x=20, line_dash="dash", line_color=COLORS['warning'],
                  annotation_text="20% of customers")
    
    fig.update_layout(
        title=title,
        xaxis_title="% of Customers (sorted by revenue)",
        yaxis_title="% of Total Revenue (cumulative)",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
    )
    
    return fig


def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Distribution plots for Recency, Frequency, and Monetary features.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Recency (days)", "Frequency (repeat purchases)", f"Monetary ({CURRENCY_SYMBOL})"],
    )
    
    fig.add_trace(
        go.Histogram(x=rfm_df['recency'], nbinsx=50, 
                     marker_color=COLORS['primary'], opacity=0.8,
                     name='Recency'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=rfm_df['frequency'], nbinsx=50,
                     marker_color=COLORS['secondary'], opacity=0.8,
                     name='Frequency'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=rfm_df['monetary_value'], nbinsx=50,
                     marker_color=COLORS['success'], opacity=0.8,
                     name='Monetary'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="RFM Feature Distributions",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=400,
    )
    
    return fig


def plot_cohort_retention_heatmap(retention_matrix: pd.DataFrame) -> go.Figure:
    """
    Cohort retention heatmap.
    
    Parameters
    ----------
    retention_matrix : pd.DataFrame
        Rows = cohort months, columns = period index, values = retention %
    """
    # Limit to reasonable number of periods
    max_periods = min(13, retention_matrix.shape[1])
    matrix = retention_matrix.iloc[:, :max_periods]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=[f"Month {i}" for i in range(max_periods)],
        y=[str(c) for c in matrix.index],
        colorscale=[
            [0, '#1E293B'],
            [0.25, '#312E81'],
            [0.5, '#6366F1'],
            [0.75, '#818CF8'],
            [1.0, '#C7D2FE'],
        ],
        text=np.round(matrix.values, 1),
        texttemplate="%{text:.0f}%",
        textfont={"size": 10},
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Retention %"),
    ))
    
    fig.update_layout(
        title="Cohort Retention Heatmap",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort (First Purchase Month)",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(matrix) * 30),
        yaxis=dict(autorange='reversed'),
    )
    
    return fig


def plot_segment_scatter(
    predicted_clv: pd.Series,
    p_alive: pd.Series,
    segment_labels: pd.Series,
    title: str = "Customer Segments: CLV vs P(Alive)"
) -> go.Figure:
    """
    2D scatter plot of CLV vs probability alive, colored by segment.
    """
    df = pd.DataFrame({
        'Predicted CLV': predicted_clv,
        'P(Alive)': p_alive,
        'Segment': segment_labels,
    })
    
    fig = px.scatter(
        df, x='P(Alive)', y='Predicted CLV',
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.6,
        title=title,
        template=PLOTLY_TEMPLATE,
        hover_data=['Predicted CLV', 'P(Alive)', 'Segment'],
    )
    
    fig.update_layout(
        xaxis_title="Probability of Being Alive",
        yaxis_title=f"Predicted CLV ({CURRENCY_SYMBOL})",
        legend_title="Segment",
        height=600,
    )
    
    fig.update_traces(marker=dict(size=5))
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart comparing model performance metrics.
    """
    models = comparison_df.index.tolist()
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE", "RMSE", "MAPE (%)"],
    )
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    
    for i, metric in enumerate(['MAE', 'RMSE', 'MAPE']):
        if metric in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=comparison_df[metric],
                    marker_color=colors[i],
                    name=metric,
                    text=comparison_df[metric].round(1),
                    textposition='auto',
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title="Model Performance Comparison",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=400,
    )
    
    return fig


def plot_roi_curve(
    budgets: np.ndarray,
    revenues: Dict[str, np.ndarray],
    title: str = "Marketing ROI Curve"
) -> go.Figure:
    """
    ROI curve showing budget vs expected incremental revenue.
    
    Parameters
    ----------
    budgets : array
        Budget levels (x-axis)
    revenues : dict
        {scenario_name: revenue_array} for each scenario
    """
    fig = go.Figure()
    
    scenario_colors = {
        'Conservative': COLORS['warning'],
        'Base': COLORS['primary'],
        'Optimistic': COLORS['success'],
    }
    
    for scenario_name, revenue in revenues.items():
        color = scenario_colors.get(scenario_name, COLORS['info'])
        fig.add_trace(go.Scatter(
            x=budgets,
            y=revenue,
            mode='lines',
            name=scenario_name,
            line=dict(color=color, width=3 if scenario_name == 'Base' else 2),
        ))
    
    # Add break-even line
    fig.add_trace(go.Scatter(
        x=budgets, y=budgets,
        mode='lines',
        name='Break-Even',
        line=dict(color=COLORS['danger'], width=1, dash='dash'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f"Marketing Budget ({CURRENCY_SYMBOL})",
        yaxis_title=f"Expected Incremental Revenue ({CURRENCY_SYMBOL})",
        template=PLOTLY_TEMPLATE,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def plot_psi_heatmap(psi_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of PSI values across features and time periods.
    
    Parameters
    ----------
    psi_df : pd.DataFrame
        Rows = features, columns = time periods, values = PSI
    """
    # Color scale: green (stable) → yellow (moderate) → red (significant)
    fig = go.Figure(data=go.Heatmap(
        z=psi_df.values,
        x=psi_df.columns.tolist(),
        y=psi_df.index.tolist(),
        colorscale=[
            [0, '#10B981'],     # Green (stable)
            [0.5, '#F59E0B'],   # Yellow (moderate)
            [1.0, '#EF4444'],   # Red (significant)
        ],
        zmin=0, zmax=0.4,
        text=np.round(psi_df.values, 3),
        texttemplate="%{text:.3f}",
        textfont={"size": 11},
        hovertemplate="Feature: %{y}<br>Period: %{x}<br>PSI: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title="PSI",
            tickvals=[0, 0.1, 0.2, 0.3, 0.4],
            ticktext=["0 (Stable)", "0.1", "0.2 (Drift)", "0.3", "0.4+"],
        ),
    ))
    
    fig.update_layout(
        title="Feature Drift Monitor (PSI)",
        xaxis_title="Time Period",
        yaxis_title="Feature",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(psi_df) * 40),
    )
    
    return fig


def plot_clv_distribution_comparison(
    baseline_clv: pd.Series,
    current_clv: pd.Series,
    title: str = "CLV Distribution: Baseline vs Current"
) -> go.Figure:
    """
    Overlaid histogram comparing CLV distributions.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=baseline_clv, name='Baseline',
        marker_color=COLORS['primary'], opacity=0.6,
        nbinsx=50,
    ))
    
    fig.add_trace(go.Histogram(
        x=current_clv, name='Current',
        marker_color=COLORS['warning'], opacity=0.6,
        nbinsx=50,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f"Predicted CLV ({CURRENCY_SYMBOL})",
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        barmode='overlay',
        height=400,
    )
    
    return fig


def plot_uplift_distribution(uplift_scores: pd.Series) -> go.Figure:
    """Distribution of uplift scores with persuadable region highlighted."""
    
    threshold = uplift_scores.quantile(0.9)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=uplift_scores[uplift_scores < threshold],
        name='Non-Persuadable',
        marker_color=COLORS['info'],
        opacity=0.7,
        nbinsx=50,
    ))
    
    fig.add_trace(go.Histogram(
        x=uplift_scores[uplift_scores >= threshold],
        name='Persuadable (Top 10%)',
        marker_color=COLORS['success'],
        opacity=0.8,
        nbinsx=20,
    ))
    
    fig.update_layout(
        title="Uplift Score Distribution",
        xaxis_title="Uplift Score",
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        barmode='overlay',
        height=400,
    )
    
    return fig


def plot_conformal_intervals(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_lower: pd.Series,
    y_upper: pd.Series,
    n_samples: int = 50,
    title: str = "CLV Predictions with Conformal Intervals"
) -> go.Figure:
    """
    Plot predicted CLV with confidence intervals vs actuals.
    Shows a sample of customers sorted by predicted CLV.
    """
    # Sample and sort
    idx = np.random.RandomState(42).choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    idx = idx[np.argsort(y_pred.iloc[idx] if hasattr(y_pred, 'iloc') else y_pred[idx])]
    
    x = np.arange(len(idx))
    
    fig = go.Figure()
    
    # Confidence interval band
    y_l = y_lower.iloc[idx] if hasattr(y_lower, 'iloc') else y_lower[idx]
    y_u = y_upper.iloc[idx] if hasattr(y_upper, 'iloc') else y_upper[idx]
    y_p = y_pred.iloc[idx] if hasattr(y_pred, 'iloc') else y_pred[idx]
    y_t = y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_u, y_l[::-1]]),
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(99, 102, 241, 0)'),
        name=f'{int((1-CONFORMAL_ALPHA)*100)}% Interval',
    ))
    
    # Predicted
    fig.add_trace(go.Scatter(
        x=x, y=y_p,
        mode='lines',
        name='Predicted CLV',
        line=dict(color=COLORS['primary'], width=2),
    ))
    
    # Actual
    fig.add_trace(go.Scatter(
        x=x, y=y_t,
        mode='markers',
        name='Actual CLV',
        marker=dict(color=COLORS['success'], size=6, opacity=0.8),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Customer (sorted by predicted CLV)",
        yaxis_title=f"CLV ({CURRENCY_SYMBOL})",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    
    return fig
