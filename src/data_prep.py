"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Prepare the modelling data. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# Columns that LEAK the target (derived from price). Drop these before training.
# Target-derived columns found in the supplied dataset.
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
]


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    from pathlib import Path
    import pandas as pd
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. See START.md.")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    import pandas as pd
    import re
    raw = df
    NUMERIC = ["area", "rooms", "floor", "total_floors", "land_area", "lat", "lng"]
    CATEGORICAL = ["location", "city", "building_type", "category", "repair", "bill_of_sale", "mortgage"]
    ALIASES = {"area": "Sahə", "rooms": "Otaq sayı", "land_area": "Torpaq sahəsi", "building_type": "Binanın növü", "category": "Kateqoriya", "repair": "Təmir", "bill_of_sale": "Çıxarış", "mortgage": "İpoteka"}
    def parse_number(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        text = re.sub(r"[\s\u00a0]", "", str(value)).replace(",", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else np.nan
    if 'price' not in raw or not any((c in raw for c in ['Sahə', 'area'])):
        raise ValueError('The dataset must contain price and area (Sahə) columns.')
    df = raw.drop(columns=LEAKAGE_COLUMNS, errors="ignore").copy()
    audit = {'raw_rows': len(df)}
    df = df.drop_duplicates()
    audit['exact_duplicates_removed'] = len(raw) - len(df)
    for key in ['estate_rel_url', 'estate_rel_url_x', 'estate_rel_url_y']:
        if key in df:
            duplicate = df[key].notna() & df[key].duplicated(keep='first')
            audit['listing_duplicates_removed'] = int(duplicate.sum())
            df = df.loc[~duplicate].copy()
            break
    out = pd.DataFrame(index=df.index)
    out['price'] = df['price'].map(parse_number)
    currency_col = next((c for c in ['currency_x', 'currency', 'currency_y'] if c in df), None)
    if currency_col:
        currency = df[currency_col].fillna('').astype(str).str.strip().str.upper()
        valid_currency = currency.isin(['AZN', '₼'])
    else:
        raise ValueError('A currency column is required; currency must not be guessed.')
    valid_price = np.isfinite(out['price']) & (out['price'] > 0)
    audit['invalid_price_rows_removed'] = int((~valid_price).sum())
    audit['non_azn_or_unknown_currency_rows_removed'] = int((valid_price & ~valid_currency).sum())
    for name in NUMERIC + CATEGORICAL:
        source = ALIASES.get(name, name)
        if source not in df:
            source = name
        series = df[source] if source in df else pd.Series(np.nan, index=df.index)
        out[name] = series.map(parse_number) if name in NUMERIC else series
    floor_source = df['Mərtəbə'] if 'Mərtəbə' in df else pd.Series('', index=df.index)
    floors = floor_source.fillna('').astype(str).str.extract('(\\d+)\\s*/\\s*(\\d+)')
    if floors[0].notna().any():
        out['floor'] = pd.to_numeric(floors[0], errors='coerce')
        out['total_floors'] = pd.to_numeric(floors[1], errors='coerce')
    area_source = df['Sahə'] if 'Sahə' in df else df['area']
    is_sot = area_source.astype(str).str.contains('sot', case=False, na=False)
    out.loc[is_sot, 'area'] *= 100
    if 'Torpaq sahəsi' in df:
        out['land_area'] *= 100
    for name in NUMERIC:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)
    for name in ['area', 'rooms', 'land_area', 'total_floors']:
        out.loc[out[name] <= 0, name] = np.nan
    out.loc[(out['floor'] < 0) | (out['floor'] > out['total_floors']), 'floor'] = np.nan
    out.loc[~out['lat'].between(-90, 90), 'lat'] = np.nan
    out.loc[~out['lng'].between(-180, 180), 'lng'] = np.nan
    for name in CATEGORICAL:
        out[name] = out[name].fillna('missing').astype(str).str.strip().str.lower().replace('', 'missing')
    out = out.loc[valid_price & valid_currency].reset_index(drop=True)
    audit['clean_rows'] = len(out)
    used = {'price', 'Mərtəbə', currency_col, *NUMERIC, *CATEGORICAL, *ALIASES.values()}
    audit['excluded_columns'] = sorted(set(raw.columns) - used)
    audit['numeric_missing_counts'] = {c: int(out[c].isna().sum()) for c in NUMERIC}
    audit['rules'] = ['Keep the first observed snapshot of each listing URL in file order.', 'Keep positive AZN prices; do not trim the target distribution.', 'Convert area units to square meters; impossible predictors become missing.', 'Use only the declared numeric and categorical predictors.', 'Fit numeric 1%/99% clipping bounds, medians and categories on training rows only.', 'Treat absent flags as missing, not as an observed negative answer.']
    if len(out) < 30:
        raise ValueError('At least 30 valid, distinct listings are required.')
    out.attrs["cleaning_audit"] = audit
    return out


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    numeric = ["area", "rooms", "floor", "total_floors", "land_area", "lat", "lng"]
    categorical = ["location", "city", "building_type", "category", "repair", "bill_of_sale", "mortgage"]
    values = df[numeric].to_numpy(dtype=float)
    state = df.attrs.get("preprocessing")
    if state is None:
        state = {
            "medians": np.array([np.nanmedian(v) if np.isfinite(v).any() else 0.0 for v in values.T]),
            "lower": np.array([np.nanquantile(v, .01) if np.isfinite(v).any() else 0.0 for v in values.T]),
            "upper": np.array([np.nanquantile(v, .99) if np.isfinite(v).any() else 0.0 for v in values.T]),
            "categories": {c: sorted(df[c].astype(str).unique()) for c in categorical},
        }
        df.attrs["preprocessing"] = state
    # Reuse training attrs for held-out rows; never refit their statistics.
    missing = ~np.isfinite(values)
    values = np.where(missing, state["medians"], values)
    parts = [np.clip(values, state["lower"], state["upper"]), missing.astype(float)]
    names = numeric + [f"{c}_missing" for c in numeric]
    for c in categorical:
        categories = np.asarray(state["categories"][c])
        parts.append((df[c].astype(str).to_numpy()[:, None] == categories[None, :]).astype(float))
        names.extend([f"{c}={value}" for value in categories])
    return np.column_stack(parts), df["price"].to_numpy(dtype=float), names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    y_price = np.asarray(y_price, dtype=float)
    if y_price.ndim != 1 or not len(y_price) or (not np.isfinite(y_price).all()):
        raise ValueError('Prices must be a nonempty finite vector.')
    threshold = float(np.median(y_price)) if threshold is None else float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite.')
    return ((y_price > threshold).astype(int), threshold)


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(X) != len(y) or len(y) < 10:
        raise ValueError("Expected matching X and y with at least ten rows.")
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("Invalid validation/test proportions.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    train_ids = order[:int((1 - val_size - test_size) * len(y))]
    remainder = order[len(train_ids):]
    if set(np.unique(y)).issubset({0, 1}):
        # For pre-existing binary targets, stratify all three partitions.
        train_ids, val_ids, test_ids = [], [], []
        for label in [0, 1]:
            ids = rng.permutation(np.flatnonzero(y == label))
            nval, ntest = max(1, int(len(ids) * val_size)), max(1, int(len(ids) * test_size))
            if len(ids) <= nval + ntest:
                raise ValueError("Too few examples per class.")
            val_ids.extend(ids[:nval]); test_ids.extend(ids[nval:nval + ntest])
            train_ids.extend(ids[nval + ntest:])
    else:
        # Reserve training rows before deriving the price-tier boundary.
        _, threshold = make_tier_label(y[train_ids])
        labels, _ = make_tier_label(y[remainder], threshold)
        val_ids, test_ids = [], []
        for label in [0, 1]:
            ids = remainder[labels == label]
            if len(ids) < 2:
                raise ValueError("Both held-out tiers need at least two rows.")
            cut = max(1, min(len(ids) - 1, int(len(ids) * val_size / (val_size + test_size))))
            val_ids.extend(ids[:cut]); test_ids.extend(ids[cut:])
    def take(ids):
        ids = np.asarray(ids, dtype=int)
        return X.iloc[ids].copy() if hasattr(X, "iloc") else np.asarray(X)[ids]
    return (take(train_ids), y[train_ids], take(val_ids), y[val_ids], take(test_ids), y[test_ids])


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    X_tr = np.asarray(X_tr, dtype=float)
    if X_tr.ndim != 2 or not len(X_tr) or not np.isfinite(X_tr).all():
        raise ValueError("Expected a finite nonempty training matrix.")
    mean, scale = X_tr.mean(axis=0), X_tr.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    result = []
    for X in (X_tr, *others):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != X_tr.shape[1] or not np.isfinite(X).all():
            raise ValueError("Incompatible feature matrix.")
        result.append((X - mean) / scale)
    return tuple(result)
