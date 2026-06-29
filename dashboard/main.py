import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Hidden Business Cards Dashboard",
    page_icon="💳",
    layout="wide",
)

# ============================================================
# EDIT THIS SECTION
# ============================================================

# Put the prepared CSV file in the same folder as this script,
# or replace this with an absolute path.
DATA_PATH = Path(__file__).parent / "dashboard_data.csv"

# Main columns used by the dashboard.
# Replace these values with your actual CSV column names.
# If a value is None, the app will try to guess it from COLUMN_CANDIDATES.
COLUMN_MAP = {
    "card_id": "card_number",
    "probability": "probability",
    "uncertainty": "uncertainty",
    "transaction_count": "transaction_count",
    "turnover": "total_turnover",
}

# Fallback aliases. The app uses these only if COLUMN_MAP value is None
# or if the exact name from COLUMN_MAP is not found in the CSV.
COLUMN_CANDIDATES = {
    "card_id": ["card_number", "card_id", "card", "client_id", "customer_id"],
    "probability": [
        "prediction_probability",
        "probability",
        "predicted_probability",
        "hidden_business_probability",
        "p_hidden_business",
        "proba",
        "prob",
        "prediction",
        "pred_mean",
        "mean_prediction",
    ],
    "uncertainty": [
        "uncertainty",
        "model_uncertainty",
        "prediction_uncertainty",
        "variance",
        "var",
        "std",
        "std_dev",
        "entropy",
    ],
    "transaction_count": [
        "transaction_count",
        "n_transactions",
        "transactions",
        "txn_count",
        "cnt_transactions",
        "num_transactions",
        "count_transactions",
    ],
    "turnover": [
        "total_turnover",
        "turnover",
        "total_amount",
        "transaction_amount_kzt",
        "sum_amount",
        "amount_sum",
        "log_turnover",
        "log_total_turnover",
    ],
}

# Columns used for automatic explanation tags if they exist in the CSV.
REASON_CODE_COLUMNS = [
    ("online_share", "High online share"),
    ("recurring_ratio", "High recurring ratio"),
    ("tokenized_ratio", "High tokenized share"),
    ("foreign_share", "High foreign share"),
    ("business_mcc_share", "High business-like MCC share"),
    ("merchant_diversity", "High merchant diversity"),
    ("mcc_diversity", "High MCC diversity"),
    ("country_diversity", "High country diversity"),
    ("avg_transaction_amount", "High average transaction amount"),
    ("average_transaction_amount", "High average transaction amount"),
    ("total_turnover", "High turnover"),
    ("turnover", "High turnover"),
    ("transaction_count", "High transaction count"),
    ("n_transactions", "High transaction count"),
]

# To keep the probability-vs-uncertainty chart responsive on large datasets.
MAX_SCATTER_POINTS = 20_000

# ============================================================
# HELPERS
# ============================================================


def guess_column(columns, candidates):
    """Return the first exact or partial candidate match from available columns."""
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in columns:
            if candidate_lower in str(col).lower():
                return col

    return None


def resolve_column(df, logical_name, required=False):
    """Resolve logical dashboard column name to actual CSV column."""
    columns = list(df.columns)
    configured = COLUMN_MAP.get(logical_name)

    if configured is not None and configured in columns:
        return configured

    guessed = guess_column(columns, COLUMN_CANDIDATES.get(logical_name, []))
    if guessed is not None:
        return guessed

    if required:
        st.error(
            f"Required column for `{logical_name}` was not found. "
            "Edit COLUMN_MAP at the top of the script."
        )
        st.write("Available CSV columns:")
        st.code("\n".join(map(str, columns)))
        st.stop()

    return None


@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)


def numeric_series(df, col, default=0.0):
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def assign_group(p, u, probability_threshold, uncertainty_threshold):
    if pd.isna(p):
        return "No prediction"
    if p >= probability_threshold and u <= uncertainty_threshold:
        return "Confident hidden business"
    if p >= probability_threshold and u > uncertainty_threshold:
        return "Uncertain hidden business"
    return "Likely consumer"


def build_reason_codes(row, reason_specs, probability_threshold, uncertainty_threshold):
    reasons = []

    if row["_p"] >= probability_threshold and row["_u"] <= uncertainty_threshold:
        reasons.append("High probability, low uncertainty")
    elif row["_p"] >= probability_threshold and row["_u"] > uncertainty_threshold:
        reasons.append("High probability, high uncertainty")

    for col, label, cutoff in reason_specs:
        value = row.get(col, np.nan)
        if pd.notna(value) and value >= cutoff:
            reasons.append(label)

    return ", ".join(reasons[:5]) if reasons else "No strong reason code"


def style_group_rows(row):
    group = row.get("prediction_group", "")
    if group == "Confident hidden business":
        return ["background-color: #ffd6d6; color: #1f2937;"] * len(row)
    if group == "Uncertain hidden business":
        return ["background-color: #fff1b8; color: #1f2937;"] * len(row)
    if group == "Likely consumer":
        return ["background-color: #d9f7d9; color: #1f2937;"] * len(row)
    return [""] * len(row)


def safe_mean(series):
    value = series.mean()
    return value if pd.notna(value) else 0.0


# ============================================================
# LOAD DATA
# ============================================================

st.title("Hidden Business Cards Dashboard")
st.caption(
    "Card-level predictions, uncertainty, review queue, and analyst-friendly reason codes."
)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"CSV file was not found: `{DATA_PATH}`. Put the CSV next to this script "
        "or edit DATA_PATH at the top of the file."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not read `{DATA_PATH}`. Error: {e}")
    st.stop()

if df.empty:
    st.warning(
        "Dataset is empty. A dashboard with no rows is just a premium blank rectangle."
    )
    st.stop()

columns = list(df.columns)

card_col = resolve_column(df, "card_id", required=False)
prob_col = resolve_column(df, "probability", required=True)
uncertainty_col = resolve_column(df, "uncertainty", required=False)
txn_col = resolve_column(df, "transaction_count", required=False)
turnover_col = resolve_column(df, "turnover", required=False)

with st.sidebar:
    st.header("Dataset")
    st.write(f"Loaded file: `{DATA_PATH}`")
    st.write(f"Rows: `{len(df):,}`")
    st.write(f"Columns: `{len(df.columns):,}`")

    st.header("Resolved columns")
    resolved_columns = pd.DataFrame(
        {
            "dashboard_role": [
                "card_id",
                "probability",
                "uncertainty",
                "transaction_count",
                "turnover",
            ],
            "csv_column": [card_col, prob_col, uncertainty_col, txn_col, turnover_col],
        }
    )
    st.dataframe(resolved_columns, hide_index=True, use_container_width=True)

    st.header("Thresholds")
    probability_threshold = st.slider(
        "Hidden business probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.19,
        step=0.001,
        format="%.3f",
    )

    scale_uncertainty = st.checkbox(
        "Scale uncertainty to [0, 1] using percentile rank",
        value=True,
        help="Useful when uncertainty is variance/std and not naturally bounded by 0 and 1.",
    )

    uncertainty_threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        format="%.2f",
    )

    top_n = st.slider("Rows in review queue", 20, 1000, 100, step=20)

# ============================================================
# PREPARE DATA
# ============================================================

data = df.copy()
data["_p"] = numeric_series(data, prob_col).clip(0, 1)

raw_uncertainty = numeric_series(data, uncertainty_col, default=0.0)
if uncertainty_col is None:
    data["_u"] = 0.0
elif scale_uncertainty:
    data["_u"] = raw_uncertainty.rank(pct=True).fillna(0.0)
else:
    data["_u"] = raw_uncertainty.clip(lower=0, upper=1).fillna(0.0)

data["prediction_group"] = [
    assign_group(p, u, probability_threshold, uncertainty_threshold)
    for p, u in zip(data["_p"], data["_u"])
]

if txn_col is not None:
    data["_transactions"] = numeric_series(data, txn_col).fillna(0).clip(lower=0)
else:
    data["_transactions"] = 1.0

if turnover_col is not None:
    data["_turnover"] = numeric_series(data, turnover_col).fillna(0).clip(lower=0)
else:
    data["_turnover"] = 0.0

data["priority_score"] = data["_p"] * (1 - data["_u"]) * np.log1p(data["_transactions"])

reason_specs = []
for col, label in REASON_CODE_COLUMNS:
    if col in data.columns:
        s = pd.to_numeric(data[col], errors="coerce")
        cutoff = s.quantile(0.90)
        if pd.notna(cutoff):
            reason_specs.append((col, label, cutoff))

data["reason_codes"] = data.apply(
    build_reason_codes,
    axis=1,
    reason_specs=reason_specs,
    probability_threshold=probability_threshold,
    uncertainty_threshold=uncertainty_threshold,
)

data["prediction_probability"] = data["_p"]
data["uncertainty_scaled"] = data["_u"]

# ============================================================
# OVERVIEW KPIS
# ============================================================

st.subheader("Overview")

total_cards = len(data)
expected_hidden = data["_p"].sum(skipna=True)
flagged = data[data["_p"] >= probability_threshold]
confident = data[data["prediction_group"] == "Confident hidden business"]
uncertain = data[data["prediction_group"] == "Uncertain hidden business"]
likely_consumer = data[data["prediction_group"] == "Likely consumer"]
confident_ratio = len(confident) / total_cards
uncertain_ratio = len(uncertain) / total_cards

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total cards", f"{total_cards:,.0f}")
kpi2.metric("Expected hidden businesses", f"{expected_hidden:,.1f}")
kpi3.metric("Flagged cards", f"{len(flagged):,.0f}")
kpi4.metric(
    "Confident hidden businesses", f"{len(confident):,.0f} ({confident_ratio:.2%})"
)
kpi5.metric(
    "Uncertain hidden businesses", f"{len(uncertain):,.0f} ({uncertain_ratio:.2%})"
)

kpi6, kpi7, kpi8 = st.columns(3)
if txn_col is not None:
    kpi6.metric(
        "Avg transactions, flagged", f"{safe_mean(flagged['_transactions']):,.2f}"
    )
else:
    kpi6.metric("Avg transactions, flagged", "n/a")

if turnover_col is not None:
    flagged_turnover = flagged["_turnover"].sum()
    all_turnover = data["_turnover"].sum()
    share = flagged_turnover / all_turnover if all_turnover > 0 else np.nan
    kpi7.metric("Total turnover, flagged", f"{flagged_turnover:,.0f}")
    kpi8.metric("Flagged turnover share", f"{share:.2%}" if pd.notna(share) else "n/a")
else:
    kpi7.metric("Total turnover, flagged", "n/a")
    kpi8.metric("Flagged turnover share", "n/a")

# ============================================================
# CHARTS
# ============================================================

left, right = st.columns([1, 2])

with left:
    st.subheader("Prediction groups")
    group_counts = (
        data["prediction_group"]
        .value_counts()
        .rename_axis("prediction_group")
        .reset_index(name="cards")
    )
    fig_group = px.pie(
        group_counts,
        names="prediction_group",
        values="cards",
        hole=0.55,
        title="Card split by prediction group",
    )
    st.plotly_chart(fig_group, use_container_width=True)

with right:
    st.subheader("Probability vs uncertainty")
    if len(data) > MAX_SCATTER_POINTS:
        scatter_data = data.sample(MAX_SCATTER_POINTS, random_state=42)
        st.caption(
            f"Showing a random sample of {MAX_SCATTER_POINTS:,} cards for speed."
        )
    else:
        scatter_data = data

    size_col = "_transactions" if txn_col is not None else None
    hover_cols = ["prediction_group", "priority_score", "reason_codes"]
    if card_col is not None:
        hover_cols = [card_col] + hover_cols
    if txn_col is not None:
        hover_cols.append(txn_col)
    if turnover_col is not None:
        hover_cols.append(turnover_col)

    fig_scatter = px.scatter(
        scatter_data,
        x="_p",
        y="_u",
        color="prediction_group",
        size=size_col,
        hover_data=hover_cols,
        labels={
            "_p": "Prediction probability",
            "_u": "Uncertainty, scaled",
            "prediction_group": "Prediction group",
        },
        title="Suspicion vs uncertainty",
        opacity=0.75,
    )
    fig_scatter.add_vline(x=probability_threshold, line_dash="dash")
    fig_scatter.add_hline(y=uncertainty_threshold, line_dash="dash")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# ANALYST REVIEW QUEUE
# ============================================================

st.subheader("Analyst review queue")

queue_filter_left, queue_filter_right = st.columns([1, 3])
with queue_filter_left:
    selected_groups = st.multiselect(
        "Groups to show",
        options=sorted(data["prediction_group"].unique()),
        default=["Confident hidden business", "Uncertain hidden business"],
    )

with queue_filter_right:
    search_text = st.text_input(
        "Search card ID",
        placeholder="Type card number/id if needed",
        disabled=card_col is None,
    )

queue = data[data["prediction_group"].isin(selected_groups)].copy()
if card_col is not None and search_text.strip():
    queue = queue[
        queue[card_col]
        .astype(str)
        .str.contains(search_text.strip(), case=False, na=False)
    ]

queue = queue.sort_values(
    ["priority_score", "_p", "_u"],
    ascending=[False, False, True],
).head(top_n)

queue_cols = []
if card_col is not None:
    queue_cols.append(card_col)
queue_cols += [
    "prediction_group",
    "prediction_probability",
    "uncertainty_scaled",
    "priority_score",
]
if txn_col is not None:
    queue_cols.append(txn_col)
if turnover_col is not None:
    queue_cols.append(turnover_col)
queue_cols.append("reason_codes")

queue_view = queue[queue_cols].copy()

st.dataframe(
    queue_view.style.apply(style_group_rows, axis=1).format(
        {
            "prediction_probability": "{:.2%}",
            "uncertainty_scaled": "{:.2%}",
            "priority_score": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

csv = queue_view.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download current review queue as CSV",
    data=csv,
    file_name="hidden_business_review_queue.csv",
    mime="text/csv",
)

# ============================================================
# GROUP COMPARISON
# ============================================================

st.subheader("Behavior comparison by prediction group")

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
exclude_internal = {
    "_p",
    "_u",
    "_transactions",
    "_turnover",
    "priority_score",
    "prediction_probability",
    "uncertainty_scaled",
}

default_features = [
    c
    for c in numeric_cols
    if c not in exclude_internal
    and any(
        key in c.lower()
        for key in [
            "online",
            "recurring",
            "turnover",
            "amount",
            "transaction",
            "mcc",
            "country",
            "merchant",
            "token",
        ]
    )
]

selected_numeric = st.multiselect(
    "Features to compare",
    options=[c for c in numeric_cols if c not in exclude_internal],
    default=default_features[:8],
)

if selected_numeric:
    comparison = data.groupby("prediction_group")[selected_numeric].mean().T
    st.dataframe(comparison, use_container_width=True)

    feature_for_chart = st.selectbox("Feature chart", selected_numeric)
    fig_feature = px.box(
        data,
        x="prediction_group",
        y=feature_for_chart,
        color="prediction_group",
        points=False,
        title=f"{feature_for_chart} by prediction group",
    )
    st.plotly_chart(fig_feature, use_container_width=True)
else:
    st.info("Select at least one numeric feature to compare groups.")

# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

st.subheader("Threshold sensitivity")

thresholds = np.linspace(0.0, 1.0, 101)
rows = []
for t in thresholds:
    tmp_flagged = data[data["_p"] >= t]
    row = {
        "threshold": t,
        "flagged_cards": len(tmp_flagged),
        "expected_hidden_businesses_above_threshold": tmp_flagged["_p"].sum(),
    }
    if turnover_col is not None:
        row["flagged_turnover"] = tmp_flagged["_turnover"].sum()
        total_turnover = data["_turnover"].sum()
        row["flagged_turnover_share"] = (
            row["flagged_turnover"] / total_turnover if total_turnover > 0 else np.nan
        )
    rows.append(row)

sensitivity = pd.DataFrame(rows)

sens_left, sens_right = st.columns(2)
with sens_left:
    fig_thr_count = px.line(
        sensitivity,
        x="threshold",
        y="flagged_cards",
        title="Analyst workload by probability threshold",
        labels={"flagged_cards": "Cards above threshold"},
    )
    fig_thr_count.add_vline(x=probability_threshold, line_dash="dash")
    st.plotly_chart(fig_thr_count, use_container_width=True)

with sens_right:
    if turnover_col is not None:
        fig_thr_turnover = px.line(
            sensitivity,
            x="threshold",
            y="flagged_turnover_share",
            title="Turnover captured by threshold",
            labels={"flagged_turnover_share": "Share of total turnover"},
        )
        fig_thr_turnover.add_vline(x=probability_threshold, line_dash="dash")
        st.plotly_chart(fig_thr_turnover, use_container_width=True)
    else:
        fig_thr_expected = px.line(
            sensitivity,
            x="threshold",
            y="expected_hidden_businesses_above_threshold",
            title="Expected hidden businesses above threshold",
        )
        fig_thr_expected.add_vline(x=probability_threshold, line_dash="dash")
        st.plotly_chart(fig_thr_expected, use_container_width=True)

# ============================================================
# DATA QUALITY
# ============================================================

st.subheader("Data quality checks")

missing = data[columns].isna().mean().sort_values(ascending=False).reset_index()
missing.columns = ["column", "missing_share"]

quality_left, quality_right = st.columns(2)
with quality_left:
    st.write("Missing values")
    st.dataframe(missing.head(20), use_container_width=True, hide_index=True)

with quality_right:
    st.write("Prediction summary")
    summary = data[["_p", "_u", "priority_score"]].describe().T
    summary = summary.rename(
        index={"_p": "prediction_probability", "_u": "uncertainty_scaled"}
    )
    st.dataframe(summary, use_container_width=True)

if txn_col is not None:
    low_activity_cutoff = st.number_input(
        "Low-activity cutoff, transaction count",
        min_value=0.0,
        value=3.0,
        step=1.0,
    )
    low_activity = data[data["_transactions"] <= low_activity_cutoff]
    st.info(
        f"Low-activity cards: {len(low_activity):,} "
        f"({len(low_activity) / len(data):.2%} of all cards). "
        "High predictions in this group should usually be checked carefully."
    )

st.caption(
    "Priority score = probability × (1 - scaled uncertainty) × log(1 + transaction_count). "
    "Treat it as a review-ranking heuristic, not as divine truth from a pie chart."
)
