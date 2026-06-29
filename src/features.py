import numpy as np
import pandas as pd
from .config import MCC_GROUPS, COUNTRIES


def date_span_days(df, date_col="transaction_date"):
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    if pd.isna(min_date) or pd.isna(max_date):
        return 1

    return int((max_date - min_date).days) + 1


def mcc_range_segment(mcc):
    x = int(mcc)

    if 3000 <= x <= 3999:
        return "travel_private"
    elif 4000 <= x <= 4799:
        return "transportation"
    elif 4800 <= x <= 4999:
        return "utilities"
    elif 5000 <= x <= 5599:
        return "retail_outlets"
    elif 5600 <= x <= 5699:
        return "clothing"
    elif 5700 <= x <= 7299:
        return "misc_stores"
    elif 7300 <= x <= 7999:
        return "business_services"
    elif 8000 <= x <= 8999:
        return "professional_membership"
    else:
        return "other"


def js_distance_matrix(P, q):
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize each card profile; empty rows stay zero.
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

    # Normalize the reference profile.
    q_sum = q.sum()
    q = q / q_sum if q_sum != 0 else q

    M = 0.5 * (P + q)

    # Compute Kullback–Leibler KL(P || M) divergence only where the log term is defined.
    kl_pm_matrix = np.zeros_like(P)
    mask_p = (P > 0) & (M > 0)
    kl_pm_matrix[mask_p] = P[mask_p] * np.log2(P[mask_p] / M[mask_p])
    kl_pm = kl_pm_matrix.sum(axis=1)

    q_matrix = np.broadcast_to(q, P.shape)

    # Compute KL(q || M) against every card profile.
    kl_qm_matrix = np.zeros_like(P)
    mask_q = (q_matrix > 0) & (M > 0)
    kl_qm_matrix[mask_q] = q_matrix[mask_q] * np.log2(q_matrix[mask_q] / M[mask_q])
    kl_qm = kl_qm_matrix.sum(axis=1)

    js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
    js_divergence = np.maximum(js_divergence, 0)

    return np.sqrt(js_divergence)


def build_all_mcc_similarity_features(
    bus,
    consumer,
    all_segments=[
        "business_services",
        "clothing",
        "misc_stores",
        "other",
        "professional_membership",
        "retail_outlets",
        "transportation",
        "travel_private",
        "utilities",
    ],
    business_profile=None,
    consumer_profile=None,
):
    bus_tmp = bus.copy()
    consumer_tmp = consumer.copy()

    bus_tmp["mcc_range_segment"] = bus_tmp["mcc"].apply(mcc_range_segment)
    consumer_tmp["mcc_range_segment"] = consumer_tmp["mcc"].apply(mcc_range_segment)

    # Use full-sample profiles unless external train-only profiles are passed.
    if business_profile is None:
        business_profile = (
            bus_tmp["mcc_range_segment"]
            .value_counts(normalize=True)
            .reindex(all_segments, fill_value=0)
        )

    if consumer_profile is None:
        consumer_profile = (
            consumer_tmp["mcc_range_segment"]
            .value_counts(normalize=True)
            .reindex(all_segments, fill_value=0)
        )

    business_mcc_features = build_mcc_similarity_features(
        bus_tmp, business_profile, consumer_profile, all_segments
    )

    consumer_mcc_features = build_mcc_similarity_features(
        consumer_tmp, business_profile, consumer_profile, all_segments
    )

    return (
        business_mcc_features,
        consumer_mcc_features,
        business_profile,
        consumer_profile,
    )


def build_mcc_similarity_features(df, business_profile, consumer_profile, all_segments):
    df = df.copy()

    counts = pd.crosstab(df["card_number"], df["mcc_range_segment"])
    counts = counts.reindex(columns=all_segments, fill_value=0)

    card_profiles = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    P = card_profiles.values

    distance_to_business = js_distance_matrix(P, business_profile.values)
    distance_to_consumer = js_distance_matrix(P, consumer_profile.values)

    features = pd.DataFrame(
        {
            "card_number": card_profiles.index,
            "mcc_business_similarity_gap": distance_to_consumer - distance_to_business,
            "mcc_avg_distance": (distance_to_consumer + distance_to_business) / 2,
        }
    )

    return features


def make_card_features_final(
    df, label, total_days, mcc_groups=MCC_GROUPS, countries=COUNTRIES
):
    """
    Build all non-MCC-similarity features at card level.

    This includes the base features, online turnover ratio, recurring-MCC mixed
    features, and country-share features. The output columns are the same as in
    dataset_no_model.parquet, except for MCC similarity columns, which are added
    by build_mcc_similarity_features().
    """

    df = df.copy()

    df["mcc_range_segment"] = df["mcc"].apply(mcc_range_segment)

    df["hour"] = df["transaction_timestamp"].dt.hour
    df["is_online"] = (df["channel"] == "online").astype(np.int8)
    df["is_foreign"] = (df["country"] != "Kazakhstan").astype(np.int8)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(np.int8)
    df["online_amount"] = np.where(
        df["is_online"] == 1, df["transaction_amount_kzt"], 0
    )

    grouped = df.groupby("card_number", sort=True)

    features = grouped.agg(
        active_days=("transaction_date", "nunique"),
        total_turnover=("transaction_amount_kzt", "sum"),
        n_transactions=("transaction_amount_kzt", "size"),
        online_share=("is_online", "mean"),
        recurring_ratio=("is_recurring", "mean"),
        online_turnover=("online_amount", "sum"),
    )

    # Same logic as x.duplicated().mean() inside each card.
    repeated_flags = grouped["transaction_amount_kzt"].transform(
        lambda x: x.duplicated()
    )
    repeated_ratio = (
        pd.DataFrame(
            {
                "card_number": df["card_number"],
                "repeated": repeated_flags.astype(np.float32),
            }
        )
        .groupby("card_number", sort=True)["repeated"]
        .mean()
    )
    features["repeated_amount_ratio"] = repeated_ratio

    features["online_turnover_ratio"] = (
        features["online_turnover"] / features["total_turnover"].replace(0, np.nan)
    ).fillna(0)

    total_turnover = grouped["transaction_amount_kzt"].sum()

    # Recurring + MCC mixed features.
    for group_name, mcc_list in mcc_groups.items():
        mcc_set = {mcc for mcc in mcc_list}
        mask = df["is_recurring"] & df["mcc"].isin(mcc_set)

        turnover = df.loc[mask].groupby("card_number")["transaction_amount_kzt"].sum()
        features[f"recurring_{group_name}_turnover_ratio"] = (
            (turnover / total_turnover.replace(0, np.nan))
            .reindex(features.index)
            .fillna(0)
        )

    features["recurring_digital_turnover_ratio"] = (
        features["recurring_advertising_turnover_ratio"]
        + features["recurring_it_services_turnover_ratio"]
        + features["recurring_subscription_turnover_ratio"]
        + features["recurring_telecom_turnover_ratio"]
    )

    # Country shares.
    for country in countries:
        mask = df["country"] == country

        country_sum = (
            df.loc[mask].groupby("card_number")["transaction_amount_kzt"].sum()
        )

        features[f"share_merchant_country_{country}"] = (
            (country_sum / features["total_turnover"]).reindex(features.index).fillna(0)
        )

    features["active_days_ratio"] = features["active_days"] / total_days
    features["turnover_per_active_day"] = (
        features["total_turnover"] / features["active_days"].replace(0, np.nan)
    ).fillna(0)
    features["tx_per_active_day"] = (
        features["n_transactions"] / features["active_days"].replace(0, np.nan)
    ).fillna(0)

    features["log_turnover_per_active_day"] = np.log1p(
        features["turnover_per_active_day"]
    )
    features["log_tx_per_active_day"] = np.log1p(features["tx_per_active_day"])

    features.drop(
        [
            "active_days",
            "tx_per_active_day",
            "turnover_per_active_day",
            "total_turnover",
            "n_transactions",
            "recurring_subscription_turnover_ratio",
            "recurring_telecom_turnover_ratio",
            "recurring_advertising_turnover_ratio",
            "online_turnover",
            "online_turnover_ratio",
        ],
        axis=1,
        inplace=True,
    )

    features = features.reset_index()

    features["type"] = label

    return features
