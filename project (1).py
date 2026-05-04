"""
Streamlit App: Media Influence Score Predictor (basic version)
Based on final_improved.ipynb
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(page_title="Influence Score Predictor", layout="wide")
st.title("Media Influence Score Predictor")

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
# Default CSV is searched for in several sensible locations so the app
# works whether it's run locally or on Streamlit Cloud.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "response.csv"),
    os.path.join(SCRIPT_DIR, "data", "response.csv"),
    os.path.join(os.getcwd(), "response.csv"),
    os.path.join(os.getcwd(), "data", "response.csv"),
    "/mount/src/mahu/response.csv",
    "response.csv",
]


def find_default_csv():
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    return None


st.sidebar.markdown("### Data source")
uploaded = st.sidebar.file_uploader(
    "Upload a different response.csv (optional)", type=["csv"]
)


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    return df


if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success("Using uploaded file.")
else:
    default_path = find_default_csv()
    if default_path is None:
        st.error(
            "**`response.csv` was not found.**\n\n"
            "The app looked in these locations and didn't find the file:\n\n"
            + "\n".join(f"- `{p}`" for p in CANDIDATE_PATHS)
            + "\n\n**To fix this:**\n"
            "1. If running on Streamlit Cloud: commit `response.csv` to the "
            "root of your GitHub repo (next to `app.py`) and redeploy.\n"
            "2. If running locally: place `response.csv` in the same folder "
            "as `app.py`.\n"
            "3. Or upload a file using the sidebar uploader."
        )
        st.stop()
    df = load_data(default_path)
    st.sidebar.info(f"Using bundled CSV:\n`{default_path}`")

st.subheader("Dataset")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())


# ---------------------------------------------------------------
# Train pipeline
# ---------------------------------------------------------------
@st.cache_resource
def train_pipeline(df):
    x = df.drop(columns="Influence_Score")
    y = pd.DataFrame(df["Influence_Score"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    encoders = {}
    cat_cols = [
        c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])
    ]

    for col in cat_cols:
        if col not in ["Gender", "Fav_Genre"]:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            encoders[col] = le
        else:
            ohe = OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )
            X_train_enc = ohe.fit_transform(X_train[[col]])
            X_test_enc = ohe.transform(X_test[[col]])
            enc_cols = ohe.get_feature_names_out([col])

            X_train_enc = pd.DataFrame(
                X_train_enc, columns=enc_cols, index=X_train.index
            )
            X_test_enc = pd.DataFrame(
                X_test_enc, columns=enc_cols, index=X_test.index
            )

            X_train = X_train.drop(columns=col).join(X_train_enc)
            X_test = X_test.drop(columns=col).join(X_test_enc)
            encoders[col] = ohe

    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Lasso feature selection
    lasso_cv = LassoCV(alphas=None, cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, Y_train.values.ravel())
    kept_features = [
        f for f, c in zip(feature_names, lasso_cv.coef_) if c != 0
    ]

    X_train_lasso = X_train_df[kept_features].reset_index(drop=True)
    X_test_lasso = X_test_df[kept_features].reset_index(drop=True)
    Y_train_reset = Y_train.reset_index(drop=True)

    # Backward elimination via p-value
    X_train_current = X_train_lasso.copy()
    X_test_current = X_test_lasso.copy()
    final_features = list(X_train_current.columns)

    while True:
        X_train_ols = sm.add_constant(X_train_current)
        model = sm.OLS(Y_train_reset, X_train_ols).fit()
        p_values = model.pvalues.drop("const")
        if p_values.max() > 0.05:
            feat = p_values.idxmax()
            X_train_current = X_train_current.drop(columns=[feat])
            X_test_current = X_test_current.drop(columns=[feat])
            final_features.remove(feat)
        else:
            break

    X_test_ols = sm.add_constant(X_test_current, has_constant="add")
    X_test_ols = X_test_ols[X_train_ols.columns]

    y_train_pred = model.predict(X_train_ols)
    y_test_pred = model.predict(X_test_ols)

    metrics = {
        "train_r2": r2_score(Y_train_reset, y_train_pred),
        "test_r2": r2_score(Y_test, y_test_pred),
        "train_mse": mean_squared_error(Y_train_reset, y_train_pred),
        "test_mse": mean_squared_error(Y_test, y_test_pred),
    }

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "cat_cols": cat_cols,
        "feature_names": feature_names,
        "final_features": final_features,
        "kept_features": kept_features,
        "metrics": metrics,
    }


results = train_pipeline(df)

# ---------------------------------------------------------------
# Show model results
# ---------------------------------------------------------------
st.subheader("Model Performance")
m = results["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train R²", f"{m['train_r2']:.4f}")
c2.metric("Test R²", f"{m['test_r2']:.4f}")
c3.metric("Train MSE", f"{m['train_mse']:.4f}")
c4.metric("Test MSE", f"{m['test_mse']:.4f}")

st.write("**Final features used by model:**", results["final_features"])

# ---------------------------------------------------------------
# Prediction form — user enters ALL original columns
# ---------------------------------------------------------------
st.subheader("Predict Influence Score")
st.write("Enter values for every column below:")

original_cols = df.drop(columns="Influence_Score").columns.tolist()

with st.form("prediction_form"):
    user_input = {}

    for col in original_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            user_input[col] = st.number_input(
                col,
                min_value=col_min,
                max_value=col_max,
                value=col_mean,
                step=1.0,
            )
        else:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            user_input[col] = st.selectbox(col, options)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])

    # Apply same encoding as training
    encoders = results["encoders"]
    for col in results["cat_cols"]:
        if col not in ["Gender", "Fav_Genre"]:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])
        else:
            ohe = encoders[col]
            enc = ohe.transform(input_df[[col]])
            enc_cols = ohe.get_feature_names_out([col])
            enc_df = pd.DataFrame(enc, columns=enc_cols, index=input_df.index)
            input_df = input_df.drop(columns=col).join(enc_df)

    # Align columns with training feature order
    feature_names = results["feature_names"]
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    scaler = results["scaler"]
    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_names)

    # Keep only final features + constant
    final_features = results["final_features"]
    model = results["model"]

    X_new = scaled_df[final_features]
    X_new = sm.add_constant(X_new, has_constant="add")
    X_new = X_new[model.model.exog_names]

    pred = float(model.predict(X_new).iloc[0])
    pred_clipped = float(np.clip(pred, 1, 10))

    st.success(f"Predicted Influence Score: {pred_clipped:.2f} / 10")
    st.caption(f"(raw model output: {pred:.2f})")