import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import RANDOM_STATE, DATA_PATHS
from .features import build_all_mcc_similarity_features


def load_data(data_paths=DATA_PATHS):
    business = pd.read_parquet(data_paths["business"])
    consumer = pd.read_parquet(data_paths["consumer"])
    merchants = pd.read_parquet(data_paths["merchants"])

    return business, consumer, merchants


def split_card_data(
    business_agg,
    consumer_agg,
    val_size=0.2,
    test_size=0.2,
    random_state=RANDOM_STATE,
    card_splits=None,
):
    business_cards = business_agg["card_number"]
    consumer_cards = consumer_agg["card_number"]

    if card_splits is None:
        business_cards_train, business_cards_tmp = train_test_split(
            business_cards, test_size=val_size + test_size, random_state=random_state
        )
        consumer_cards_train, consumer_cards_tmp = train_test_split(
            consumer_cards, test_size=val_size + test_size, random_state=random_state
        )

        business_cards_val, business_cards_test = train_test_split(
            business_cards_tmp,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
        )
        consumer_cards_val, consumer_cards_test = train_test_split(
            consumer_cards_tmp,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
        )
    else:
        business_cards_train = business_agg[
            business_agg["card_number"].isin(card_splits["business"]["train"])
        ]
        business_cards_val = business_agg[
            business_agg["card_number"].isin(card_splits["business"]["val"])
        ]
        business_cards_test = business_agg[
            business_agg["card_number"].isin(card_splits["business"]["test"])
        ]

        consumer_cards_train = consumer_agg[
            consumer_agg["card_number"].isin(card_splits["consumer"]["train"])
        ]
        consumer_cards_val = consumer_agg[
            consumer_agg["card_number"].isin(card_splits["consumer"]["val"])
        ]
        consumer_cards_test = consumer_agg[
            consumer_agg["card_number"].isin(card_splits["consumer"]["test"])
        ]

    return {
        "business": {
            "train": business_cards_train,
            "val": business_cards_val,
            "test": business_cards_test,
        },
        "consumer": {
            "train": consumer_cards_train,
            "val": consumer_cards_val,
            "test": consumer_cards_test,
        },
    }


def split_transactions_by_cards(
    business, consumer, card_splits, card_col="card_number"
):
    business_splits = {}
    consumer_splits = {}

    for split in ["train", "val", "test"]:
        business_splits[split] = business[
            business[card_col].isin(card_splits["business"][split])
        ]
        consumer_splits[split] = consumer[
            consumer[card_col].isin(card_splits["consumer"][split])
        ]

    return {"business": business_splits, "consumer": consumer_splits}


def combine_business_consumer_sets(business_splits, consumer_splits):
    combined_splits = {}

    for split in ["train", "val", "test"]:
        combined_splits[split] = pd.concat(
            [business_splits[split], consumer_splits[split]], ignore_index=True
        )

    return combined_splits


def make_xy(combined_splits, target_col="type"):
    X = {}
    y = {}

    for split in ["train", "val", "test"]:
        X[split] = combined_splits[split].drop(columns=[target_col])
        y[split] = combined_splits[split][target_col]

    return X, y


def fit_transform_scale_features(
    X, features_to_scale, suffix="_scaled", scaler=None, drop_original=True
):
    if scaler is None:
        scaler = RobustScaler()

    scaled_cols = [f"{col}{suffix}" for col in features_to_scale]

    X_scaled = X.copy()
    X_scaled[scaled_cols] = scaler.fit_transform(X[features_to_scale])

    if drop_original:
        X_scaled = X_scaled.drop(columns=features_to_scale)

    return X_scaled, scaler


def transform_scale_features(
    X, features_to_scale, scaler, suffix="_scaled", drop_original=True
):
    scaled_cols = [f"{col}{suffix}" for col in features_to_scale]

    X_scaled = X.copy()
    X_scaled[scaled_cols] = scaler.transform(X[features_to_scale])

    if drop_original:
        X_scaled = X_scaled.drop(columns=features_to_scale)

    return X_scaled


def fit_transform_mcc_similarity_features(
    card_splits, transaction_splits, all_segments, card_col="card_number"
):
    # Raw transaction splits
    bus_tx_train = transaction_splits["business"]["train"]
    bus_tx_val = transaction_splits["business"]["val"]
    bus_tx_test = transaction_splits["business"]["test"]

    consumer_tx_train = transaction_splits["consumer"]["train"]
    consumer_tx_val = transaction_splits["consumer"]["val"]
    consumer_tx_test = transaction_splits["consumer"]["test"]

    # Card-level splits
    bus_agg_train = card_splits["business"]["train"]
    bus_agg_val = card_splits["business"]["val"]
    bus_agg_test = card_splits["business"]["test"]

    consumer_agg_train = card_splits["consumer"]["train"]
    consumer_agg_val = card_splits["consumer"]["val"]
    consumer_agg_test = card_splits["consumer"]["test"]

    # Fit profiles on train only
    (
        business_mcc_features_train,
        consumer_mcc_features_train,
        business_profile,
        consumer_profile,
    ) = build_all_mcc_similarity_features(
        bus_tx_train,
        consumer_tx_train,
        all_segments,
        business_profile=None,
        consumer_profile=None,
    )

    # Transform validation using train profiles
    (
        business_mcc_features_val,
        consumer_mcc_features_val,
        _,
        _,
    ) = build_all_mcc_similarity_features(
        bus_tx_val,
        consumer_tx_val,
        all_segments,
        business_profile=business_profile,
        consumer_profile=consumer_profile,
    )

    # Transform test using train profiles
    (
        business_mcc_features_test,
        consumer_mcc_features_test,
        _,
        _,
    ) = build_all_mcc_similarity_features(
        bus_tx_test,
        consumer_tx_test,
        all_segments,
        business_profile=business_profile,
        consumer_profile=consumer_profile,
    )

    # Merge MCC features into card-level datasets
    bus_train = bus_agg_train.merge(
        business_mcc_features_train,
        on=card_col,
        how="left",
    )

    bus_val = bus_agg_val.merge(
        business_mcc_features_val,
        on=card_col,
        how="left",
    )

    bus_test = bus_agg_test.merge(
        business_mcc_features_test,
        on=card_col,
        how="left",
    )

    consumer_train = consumer_agg_train.merge(
        consumer_mcc_features_train,
        on=card_col,
        how="left",
    )

    consumer_val = consumer_agg_val.merge(
        consumer_mcc_features_val,
        on=card_col,
        how="left",
    )

    consumer_test = consumer_agg_test.merge(
        consumer_mcc_features_test,
        on=card_col,
        how="left",
    )

    updated_card_splits = {
        "business": {
            "train": bus_train,
            "val": bus_val,
            "test": bus_test,
        },
        "consumer": {
            "train": consumer_train,
            "val": consumer_val,
            "test": consumer_test,
        },
    }

    profiles = {
        "business_profile": business_profile,
        "consumer_profile": consumer_profile,
    }

    return updated_card_splits, profiles
