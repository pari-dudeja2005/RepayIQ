
import hashlib
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import REPORT_DIR
from src.predict import (
    predict_customers,
    load_model,
    prepare_model_features,
)
from src.explain import explain_customer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RepayIQ | Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# =====================================================
# CONSTANTS
# =====================================================

RISK_COLORS = {
    "Low Risk": "#22C55E",
    "Medium Risk": "#F59E0B",
    "High Risk": "#EF4444",
}

RISK_LEVELS = [
    "Low Risk",
    "Medium Risk",
    "High Risk",
]

RAW_COLUMNS = (
    ["credit_limit"]
    + [f"repay_status_{i}" for i in range(1, 7)]
    + [f"bill_{i}" for i in range(1, 7)]
    + [f"payment_{i}" for i in range(1, 7)]
)

COLUMN_MAPPING = {
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
    "Y": "default_next_month",
}


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_resource
def get_model_bundle(model_choice):
    # load_model reads REPAYIQ_MODEL from environment.
    # This is set before calling this function.
    return load_model()


@st.cache_data(show_spinner=False)
def read_demo_data():
    path = REPORT_DIR / "demo_customers.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Demo dataset not found: {path}"
        )

    data = pd.read_csv(path)

    if "Customer ID" not in data.columns:
        data.insert(
            0,
            "Customer ID",
            [
                f"DEMO-{i:05d}"
                for i in range(1, len(data) + 1)
            ],
        )

    return data


@st.cache_data(show_spinner=False)
def get_predictions(model_choice):
    data = read_demo_data()
    return predict_customers(data)


@st.cache_data(show_spinner=False)
def get_explanation(customer_id, model_choice):
    data = read_demo_data()

    record = data[
        data["Customer ID"] == customer_id
    ].head(1)

    if record.empty:
        raise ValueError(
            f"Customer not found: {customer_id}"
        )

    return explain_customer(record)


# =====================================================
# HELPERS
# =====================================================

def style_chart(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=10, r=10, t=35, b=10),
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


def risk_distribution(data):
    counts = (
        data["risk_level"]
        .value_counts()
        .reindex(RISK_LEVELS, fill_value=0)
        .reset_index()
    )

    counts.columns = [
        "Risk Level",
        "Accounts",
    ]

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Accounts",
        hole=0.55,
        color="Risk Level",
        color_discrete_map=RISK_COLORS,
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
        "payment_to_bill_ratio",
    ]

    display = data[columns].copy()

    display = display.rename(
        columns={
            "risk_percentage": "Risk Score (%)",
            "risk_level": "Risk Level",
            "credit_limit": "Credit Limit",
            "late_payment_count": "Late Payment Months",
            "recent_delay": "Recent Status",
            "payment_to_bill_ratio": "Payment-to-Bill Ratio",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


def prepare_uploaded_data(uploaded_df, bundle):
    data = uploaded_df.copy()

    if data.empty:
        raise ValueError("Uploaded CSV is empty.")

    if len(data) > 10000:
        raise ValueError(
            "Maximum batch size is 10,000 records."
        )

    if data.columns.duplicated().any():
        raise ValueError(
            "CSV contains duplicate column names."
        )

    # Support original UCI column names.
    data = data.rename(
        columns=COLUMN_MAPPING
    )

    if data.columns.duplicated().any():
        raise ValueError(
            "CSV contains conflicting column names "
            "after renaming."
        )

    # Preserve customer IDs where provided.
    if "Customer ID" not in data.columns:
        if "ID" in data.columns:
            data = data.rename(
                columns={"ID": "Customer ID"}
            )
        else:
            data.insert(
                0,
                "Customer ID",
                [
                    f"BATCH-{i:05d}"
                    for i in range(1, len(data) + 1)
                ],
            )

    ids = data["Customer ID"]

    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
        or ids.astype(str).duplicated().any()
    ):
        raise ValueError(
            "Customer IDs must be unique and non-empty."
        )

    # Remove labels and excluded demographic inputs.
    data = data.drop(
        columns=[
            "default_next_month",
            "sex",
            "education",
            "marital_status",
            "age",
        ],
        errors="ignore",
    )

    missing_raw = [
        col for col in RAW_COLUMNS
        if col not in data.columns
    ]

    if missing_raw:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_raw)
        )

    for col in RAW_COLUMNS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    values = data[RAW_COLUMNS].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "CSV contains missing or invalid "
            "numeric values."
        )

    if (data["credit_limit"] <= 0).any():
        raise ValueError(
            "Credit limit must be greater than zero."
        )

    # Validate the complete model feature pipeline.
    # predict_customers will prepare features again.
    # We return raw inputs to avoid passing modified
    # features as if they were user-provided values.
    prepare_model_features(data, bundle)

    return data


def export_dataframe(data):
    export = data.drop(
        columns=["warning_indicators"],
        errors="ignore",
    )

    return export.to_csv(index=False)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("RepayIQ")
st.sidebar.caption("Credit Risk Intelligence Platform")

model_choice = st.sidebar.selectbox(
    "Prediction Model",
    [
        "Original Model",
        "Advanced CatBoost",
    ],
    index=0,
)

model_key = (
    "catboost"
    if model_choice == "Advanced CatBoost"
    else "original"
)

os.environ["REPAYIQ_MODEL"] = model_key

try:
    bundle = get_model_bundle(model_key)
except Exception as error:
    st.error(
        f"Unable to load {model_choice}: {error}"
    )
    st.stop()

st.sidebar.divider()

st.sidebar.write(
    "**Active Model:**",
    bundle.get("model_name", model_choice),
)

st.sidebar.write(
    "**Decision Threshold:**",
    f"{float(bundle['threshold']):.2f}",
)

st.sidebar.write(
    "**Model Features:**",
    len(bundle["features"]),
)

st.sidebar.caption(
    "Risk bands are illustrative and are not "
    "approved lending decision rules."
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Monitoring",
        "Customer Analysis",
        "Batch Analysis",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "RepayIQ | Explainable Repayment Risk Analytics"
)


# =====================================================
# LOAD PREDICTIONS
# =====================================================

try:
    df = get_predictions(model_key)
except Exception as error:
    st.error(
        f"Unable to generate predictions: {error}"
    )
    st.stop()


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.title("Overview")

    st.caption(
        "Portfolio-level repayment risk monitoring"
    )

    total = len(df)

    flagged = int(
        df["predicted_default"].sum()
    )

    high = int(
        (df["risk_level"] == "High Risk").sum()
    )

    average_risk = df[
        "risk_percentage"
    ].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Accounts",
        f"{total:,}",
    )

    c2.metric(
        "Flagged Accounts",
        f"{flagged:,}",
    )

    c3.metric(
        "High Risk",
        f"{high:,}",
    )

    c4.metric(
        "Average Risk",
        f"{average_risk:.1f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")

        st.plotly_chart(
            risk_distribution(df),
            use_container_width=True,
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
            color_discrete_sequence=[
                "#3B82F6"
            ],
        )

        fig.update_layout(
            yaxis_title="Accounts"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Priority Accounts")

    priority = df[
        df["predicted_default"] == 1
    ].head(10)

    if priority.empty:
        st.info(
            "No accounts are flagged at the "
            "current decision threshold."
        )
    else:
        display_risk_table(priority)


# =====================================================
# RISK MONITORING
# =====================================================

elif page == "Risk Monitoring":

    st.title("Risk Monitoring")

    st.caption(
        "Explore, filter, and export customer risk."
    )

    selected_levels = st.multiselect(
        "Risk Level",
        [
            "High Risk",
            "Medium Risk",
            "Low Risk",
        ],
        default=[
            "High Risk",
            "Medium Risk",
            "Low Risk",
        ],
    )

    filtered = df[
        df["risk_level"].isin(selected_levels)
    ].copy()

    st.metric(
        "Matching Accounts",
        f"{len(filtered):,}",
    )

    display_risk_table(filtered)

    st.download_button(
        "Export Risk Report",
        data=export_dataframe(filtered),
        file_name="repayiq_risk_report.csv",
        mime="text/csv",
    )


# =====================================================
# CUSTOMER ANALYSIS
# =====================================================

elif page == "Customer Analysis":

    st.title("Customer Analysis")

    st.caption(
        "Customer-level risk insights and explanations"
    )

    selected_id = st.selectbox(
        "Select Customer",
        df["Customer ID"].tolist(),
    )

    customer_df = df[
        df["Customer ID"] == selected_id
    ]

    customer = customer_df.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Risk Score",
        f"{customer['risk_percentage']:.2f}%",
    )

    c2.metric(
        "Risk Level",
        customer["risk_level"],
    )

    c3.metric(
        "Late Payment Months",
        int(customer["late_payment_count"]),
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
        ],
    })

    fig = px.bar(
        history,
        x="Month",
        y=["Payment", "Bill"],
        barmode="group",
        color_discrete_map={
            "Payment": "#22C55E",
            "Bill": "#3B82F6",
        },
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Explainable AI: Risk Drivers")

    st.caption(
        "Understand which features increase or "
        "decrease the model's risk score."
    )

    if st.button(
        "Explain Prediction",
        type="primary",
    ):

        try:
            with st.spinner(
                "Calculating SHAP contributions..."
            ):
                explanation = get_explanation(
                    selected_id,
                    model_key,
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
                    "Decreases Risk": "#22C55E",
                },
            )

            fig.update_layout(
                xaxis_title="SHAP Contribution",
                yaxis_title="",
            )

            st.plotly_chart(
                style_chart(fig, height=500),
                use_container_width=True,
            )

            st.dataframe(
                explanation.head(12)[
                    ["Feature", "SHAP Value"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            if model_key == "catboost":
                st.caption(
                    "CatBoost SHAP contributions are "
                    "expressed in log-odds, not "
                    "probability percentage points."
                )
            else:
                st.caption(
                    "Original model explanations use "
                    "permutation-based SHAP."
                )

        except Exception as error:
            st.error(
                f"Explanation failed: {error}"
            )


# =====================================================
# BATCH ANALYSIS
# =====================================================

elif page == "Batch Analysis":

    st.title("Batch Analysis")

    st.caption(
        "Upload customer records for bulk risk scoring"
    )

    st.subheader("Import Data")

    try:
        sample = read_demo_data()

        sample_columns = (
            ["Customer ID"] + RAW_COLUMNS
        )

        sample = sample[
            sample_columns
        ].head(10)

        st.download_button(
            "Download Sample CSV",
            data=sample.to_csv(index=False),
            file_name="repayiq_sample.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.warning(
            f"Sample CSV unavailable: {error}"
        )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        file_signature = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        analysis_signature = (
            file_signature,
            model_key,
        )

        if (
            st.session_state.get(
                "batch_analysis_signature"
            ) != analysis_signature
        ):
            st.session_state[
                "batch_analysis_signature"
            ] = analysis_signature

            st.session_state.pop(
                "batch_results",
                None,
            )

        try:
            uploaded_df = pd.read_csv(
                uploaded_file
            )

            prepared_df = prepare_uploaded_data(
                uploaded_df,
                bundle,
            )

            st.success(
                f"{len(prepared_df):,} records loaded"
            )

            st.subheader("Data Preview")

            st.dataframe(
                prepared_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Run Analysis",
                type="primary",
            ):

                with st.spinner(
                    "Generating predictions..."
                ):
                    results = predict_customers(
                        prepared_df
                    )

                st.session_state[
                    "batch_results"
                ] = results

                st.success(
                    "Batch analysis completed!"
                )

        except Exception as error:
            st.error(str(error))

            st.session_state.pop(
                "batch_results",
                None,
            )

    if (
        uploaded_file is not None
        and "batch_results" in st.session_state
    ):

        results = st.session_state[
            "batch_results"
        ]

        st.divider()

        st.subheader("Batch Results")

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

        c1.metric(
            "Accounts Analyzed",
            f"{total:,}",
        )

        c2.metric(
            "Flagged Accounts",
            f"{flagged:,}",
        )

        c3.metric(
            "High Risk",
            f"{high:,}",
        )

        st.subheader("Risk Distribution")

        st.plotly_chart(
            risk_distribution(results),
            use_container_width=True,
        )

        st.subheader("Priority Accounts")

        display_risk_table(results)

        st.download_button(
            "Export Batch Results",
            data=export_dataframe(results),
            file_name="repayiq_batch_results.csv",
            mime="text/csv",
        )



