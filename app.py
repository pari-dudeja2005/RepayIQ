
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import REPORT_DIR
from src.predict import predict_customers, load_model
from src.explain import explain_customer
from src.feature_engineering import create_features


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RepayIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# STYLING
# =====================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #CBD5E1 !important;
        font-size: 14px !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #334155;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_resource
def get_model_bundle():
    return load_model()


@st.cache_data
def load_data():
    data = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    data.insert(
        0,
        "Customer ID",
        [
            f"DEMO-{i:05d}"
            for i in range(1, len(data) + 1)
        ]
    )

    return predict_customers(data)


@st.cache_data(show_spinner=False)
def get_explanation(customer_id):
    record = df[
        df["Customer ID"] == customer_id
    ].head(1)

    return explain_customer(record)


bundle = get_model_bundle()
required_features = bundle["features"]
df = load_data()


# =====================================================
# HELPERS
# =====================================================

RISK_COLORS = {
    "Low Risk": "#22C55E",
    "Medium Risk": "#F59E0B",
    "High Risk": "#EF4444"
}


def style_chart(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=10, r=10, t=35, b=10)
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


def risk_distribution(data):
    counts = (
        data["risk_level"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
        .reset_index()
    )

    counts.columns = [
        "Risk Level",
        "Accounts"
    ]

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Accounts",
        hole=0.55,
        color="Risk Level",
        color_discrete_map=RISK_COLORS
    )

    return style_chart(fig)


def display_risk_table(data):
    columns = [
        "Customer ID",
        "risk_percentage",
        "risk_level",
        "credit_limit",
        "late_payment_count",
        "recent_delay",
        "payment_to_bill_ratio"
    ]

    display = data[columns].rename(columns={
        "risk_percentage": "Risk Score (%)",
        "risk_level": "Risk Level",
        "credit_limit": "Credit Limit",
        "late_payment_count": "Late Payment Months",
        "recent_delay": "Recent Status",
        "payment_to_bill_ratio": "Payment-to-Bill Ratio"
    })

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )


def prepare_uploaded_data(uploaded_df):
    data = uploaded_df.copy()

    if data.empty:
        raise ValueError("CSV is empty.")

    if len(data) > 10000:
        raise ValueError(
            "Maximum batch size is 10,000 records."
        )

    if data.columns.duplicated().any():
        raise ValueError(
            "CSV contains duplicate column names."
        )

    column_mapping = {
        "X1": "credit_limit",
        "X2": "sex",
        "X3": "education",
        "X4": "marital_status",
        "X5": "age",
        **{
            f"X{i + 5}": f"repay_status_{i}"
            for i in range(1, 7)
        },
        **{
            f"X{i + 11}": f"bill_{i}"
            for i in range(1, 7)
        },
        **{
            f"X{i + 17}": f"payment_{i}"
            for i in range(1, 7)
        },
        "Y": "default_next_month"
    }

    data = data.rename(columns=column_mapping)

    data = data.drop(
        columns=[
            "default_next_month",
            "sex",
            "education",
            "marital_status",
            "age"
        ],
        errors="ignore"
    )

    missing_features = [
        col for col in required_features
        if col not in data.columns
    ]

    if missing_features:
        raw_columns = (
            ["credit_limit"]
            + [
                f"repay_status_{i}"
                for i in range(1, 7)
            ]
            + [
                f"bill_{i}"
                for i in range(1, 7)
            ]
            + [
                f"payment_{i}"
                for i in range(1, 7)
            ]
        )

        missing_raw = [
            col for col in raw_columns
            if col not in data.columns
        ]

        if missing_raw:
            raise ValueError(
                "Missing columns: "
                + ", ".join(missing_raw)
            )

        for col in raw_columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        if data[raw_columns].isna().any().any():
            raise ValueError(
                "Invalid numeric values in CSV."
            )

        data = create_features(data)

    for col in required_features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    values = data[required_features].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "CSV contains missing or invalid values."
        )

    if (data["credit_limit"] <= 0).any():
        raise ValueError(
            "Credit limit must be positive."
        )

    if "Customer ID" not in data.columns:
        data.insert(
            0,
            "Customer ID",
            [
                f"BATCH-{i:05d}"
                for i in range(1, len(data) + 1)
            ]
        )
    else:
        if (
            data["Customer ID"].isna().any()
            or data["Customer ID"].astype(str).duplicated().any()
        ):
            raise ValueError(
                "Customer IDs must be unique and non-empty."
            )

    return data


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("RepayIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Monitoring",
        "Customer Analysis",
        "Batch Analysis"
    ]
)


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.title("Overview")

    total = len(df)
    flagged = int(df["predicted_default"].sum())
    high = int(
        (df["risk_level"] == "High Risk").sum()
    )
    avg = df["risk_percentage"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Accounts", f"{total:,}")
    c2.metric("Flagged Accounts", f"{flagged:,}")
    c3.metric("High Risk", f"{high:,}")
    c4.metric("Average Risk", f"{avg:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")

        st.plotly_chart(
            risk_distribution(df),
            width="stretch"
        )

    with right:
        st.subheader("Risk Score Distribution")

        fig = px.histogram(
            df,
            x="risk_percentage",
            nbins=30,
            labels={
                "risk_percentage": "Risk Score (%)"
            },
            color_discrete_sequence=["#3B82F6"]
        )

        fig.update_layout(
            yaxis_title="Accounts"
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    st.subheader("Priority Accounts")

    priority = df[
        df["predicted_default"] == 1
    ].head(10)

    display_risk_table(priority)


# =====================================================
# RISK MONITORING
# =====================================================

elif page == "Risk Monitoring":

    st.title("Risk Monitoring")

    selected = st.multiselect(
        "Risk Level",
        ["High Risk", "Medium Risk", "Low Risk"],
        default=[
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ]
    )

    filtered = df[
        df["risk_level"].isin(selected)
    ]

    st.metric(
        "Matching Accounts",
        f"{len(filtered):,}"
    )

    display_risk_table(filtered)

    export = filtered.drop(
        columns=["warning_indicators"],
        errors="ignore"
    )

    st.download_button(
        "Export Report",
        data=export.to_csv(index=False),
        file_name="repayiq_risk_report.csv",
        mime="text/csv"
    )


# =====================================================
# CUSTOMER ANALYSIS
# =====================================================

elif page == "Customer Analysis":

    st.title("Customer Analysis")

    selected_id = st.selectbox(
        "Customer",
        df["Customer ID"].tolist()
    )

    customer_df = df[
        df["Customer ID"] == selected_id
    ]

    customer = customer_df.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Risk Score",
        f"{customer['risk_percentage']:.2f}%"
    )

    c2.metric(
        "Risk Level",
        customer["risk_level"]
    )

    c3.metric(
        "Late Payment Months",
        int(customer["late_payment_count"])
    )

    st.divider()

    st.subheader("Risk Indicators")

    for warning in customer["warning_indicators"]:
        st.warning(warning)

    st.subheader("Repayment History")

    history = pd.DataFrame({
        "Month": [
            f"Month {i}"
            for i in range(1, 7)
        ],
        "Payment": [
            customer[f"payment_{i}"]
            for i in range(1, 7)
        ],
        "Bill": [
            customer[f"bill_{i}"]
            for i in range(1, 7)
        ]
    })

    fig = px.bar(
        history,
        x="Month",
        y=["Payment", "Bill"],
        barmode="group",
        color_discrete_map={
            "Payment": "#22C55E",
            "Bill": "#3B82F6"
        }
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

    st.divider()

    st.subheader("Risk Drivers")

    if st.button(
        "Explain Prediction",
        type="primary"
    ):

        with st.spinner("Calculating..."):
            explanation = get_explanation(
                selected_id
            )

        top = explanation.head(12).copy()

        top["Impact"] = top[
            "SHAP Value"
        ].apply(
            lambda value: (
                "Increases Risk"
                if value > 0
                else "Decreases Risk"
            )
        )

        fig = px.bar(
            top.sort_values("SHAP Value"),
            x="SHAP Value",
            y="Feature",
            color="Impact",
            orientation="h",
            color_discrete_map={
                "Increases Risk": "#EF4444",
                "Decreases Risk": "#22C55E"
            }
        )

        fig.update_layout(
            xaxis_title="SHAP Contribution",
            yaxis_title=""
        )

        st.plotly_chart(
            style_chart(fig, height=500),
            width="stretch"
        )

        st.dataframe(
            explanation.head(12)[
                ["Feature", "SHAP Value"]
            ],
            width="stretch",
            hide_index=True
        )


# =====================================================
# BATCH ANALYSIS
# =====================================================

elif page == "Batch Analysis":

    st.title("Batch Analysis")

    st.subheader("Import Data")

    sample = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    sample = sample[required_features].head(10)

    st.download_button(
        "Download Sample CSV",
        data=sample.to_csv(index=False),
        file_name="repayiq_sample.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        file_signature = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        if (
            st.session_state.get(
                "batch_file_signature"
            ) != file_signature
        ):
            st.session_state[
                "batch_file_signature"
            ] = file_signature

            st.session_state.pop(
                "batch_results",
                None
            )

        try:
            uploaded_df = pd.read_csv(
                uploaded_file
            )

            prepared_df = prepare_uploaded_data(
                uploaded_df
            )

            st.success(
                f"{len(prepared_df):,} records loaded"
            )

            st.subheader("Data Preview")

            st.dataframe(
                prepared_df.head(10),
                width="stretch",
                hide_index=True
            )

            if st.button(
                "Run Analysis",
                type="primary"
            ):

                with st.spinner("Processing..."):
                    results = predict_customers(
                        prepared_df
                    )

                st.session_state[
                    "batch_results"
                ] = results

                st.success("Analysis complete")

        except Exception as error:
            st.error(str(error))

            st.session_state.pop(
                "batch_results",
                None
            )

    if "batch_results" in st.session_state:

        results = st.session_state[
            "batch_results"
        ]

        st.divider()

        st.subheader("Results")

        total = len(results)
        flagged = int(
            results["predicted_default"].sum()
        )
        high = int(
            (
                results["risk_level"] == "High Risk"
            ).sum()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Accounts Analyzed", total)
        c2.metric("Flagged Accounts", flagged)
        c3.metric("High Risk", high)

        st.subheader("Risk Distribution")

        st.plotly_chart(
            risk_distribution(results),
            width="stretch"
        )

        st.subheader("Priority Accounts")

        display_risk_table(results)

        export = results.drop(
            columns=["warning_indicators"],
            errors="ignore"
        )

        st.download_button(
            "Export Results",
            data=export.to_csv(index=False),
            file_name="repayiq_batch_results.csv",
            mime="text/csv"
        )


