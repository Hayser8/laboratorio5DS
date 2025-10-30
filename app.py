# app.py — Interactive Dashboard for Lab 10 (uses test.csv)
# --------------------------------------------------------
# How to run locally:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
#
# Minimal requirements.txt:
#   streamlit>=1.36
#   pandas>=2.2
#   numpy>=1.26
#   scikit-learn>=1.4
#   plotly>=5.22
#   pillow>=10.3
#
# Deployment:
#   • Push app.py + requirements.txt (+ test.csv opcional) a GitHub
#   • Deploy en Streamlit Community Cloud (share.streamlit.io)

import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

st.set_page_config(page_title="Lab 10 Dashboard", layout="wide")

# ------------------------
# Utilities
# ------------------------

def load_data(default_path: str = "test.csv") -> pd.DataFrame:
    up = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])
    if up is not None:
        return pd.read_csv(up)
    try:
        return pd.read_csv(default_path)
    except Exception as e:
        st.error(f"Could not load '{default_path}'. Upload a CSV to continue.\n{e}")
        return pd.DataFrame()


def split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    num_cols, cat_cols, dt_cols = [], [], []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            # try datetime
            if df[c].dtype == object:
                try:
                    pd.to_datetime(df[c])
                    dt_cols.append(c)
                    continue
                except Exception:
                    pass
            cat_cols.append(c)
    return num_cols, cat_cols, dt_cols


def coerce_datetimes(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    for c in dt_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass
    return df


def apply_global_filters(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], dt_cols: List[str]) -> pd.DataFrame:
    st.sidebar.markdown("### Global Filters")
    filtered = df.copy()

    # Categorical (limitamos a 10 para no saturar la UI)
    for c in cat_cols[:10]:
        if filtered[c].nunique() <= 200:
            vals = sorted([v for v in filtered[c].dropna().unique()])
            default_vals = vals if len(vals) <= 12 else []
            sel = st.sidebar.multiselect(f"{c}", options=vals, default=default_vals)
            if sel:
                filtered = filtered[filtered[c].isin(sel)]

    # Numeric (hasta 8 sliders)
    for c in num_cols[:8]:
        col_min, col_max = float(filtered[c].min()), float(filtered[c].max())
        if not (np.isfinite(col_min) and np.isfinite(col_max)):
            continue
        r = st.sidebar.slider(f"{c}", min_value=col_min, max_value=col_max, value=(col_min, col_max))
        filtered = filtered[(filtered[c] >= r[0]) & (filtered[c] <= r[1])]

    # Datetime (solo la primera como rango)
    if dt_cols:
        c = dt_cols[0]
        min_dt, max_dt = filtered[c].min(), filtered[c].max()
        if pd.notnull(min_dt) and pd.notnull(max_dt) and min_dt != max_dt:
            r = st.sidebar.date_input(f"Date range ({c})", value=(min_dt.date(), max_dt.date()))
            if isinstance(r, tuple) and len(r) == 2:
                start, end = pd.to_datetime(r[0]), pd.to_datetime(r[1])
                filtered = filtered[(filtered[c] >= start) & (filtered[c] <= end)]

    return filtered


def pick_target(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[Optional[str], str]:
    st.markdown("### 1) Select Target & Task")
    cols = list(df.columns)
    if not cols:
        return None, "none"
    target = st.selectbox("Choose target column", options=[None] + cols, index=0)
    if target is None:
        return None, "none"
    # inferencia simple
    inferred = "classification" if (df[target].nunique() <= max(20, len(df)*0.1) and not pd.api.types.is_numeric_dtype(df[target])) else "regression"
    task = st.radio("Task type (override if needed)", options=["classification", "regression"], index=0 if inferred=="classification" else 1, horizontal=True)
    return target, task


def build_models(task: str, rf_n_estimators: int, rf_max_depth: Optional[int], knn_k: int):
    if task == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth or None, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=knn_k)
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth or None, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=knn_k)
        }


def train_and_eval(df: pd.DataFrame, target: str, task: str, test_size: float,
                   rf_n_estimators: int, rf_max_depth: Optional[int], knn_k: int
                   ) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Dict[str, float]], Tuple[np.ndarray, np.ndarray]]:
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    models = build_models(task, rf_n_estimators, rf_max_depth, knn_k)
    chosen_models = st.multiselect("Choose models to train (pick up to 3)", options=list(models.keys()), default=list(models.keys()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if task=="classification" else None
    )

    fitted: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name in chosen_models:
        pipe = Pipeline(steps=[("prep", pre), ("model", models[name])])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if task == "classification":
            m = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
            }
        else:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            m = {"MAE": mae, "RMSE": rmse, "R2": r2}
        fitted[name] = pipe
        metrics[name] = m

    metrics_df = pd.DataFrame(metrics).T
    return metrics_df, fitted, metrics, (y_test.to_numpy(), y_pred if len(chosen_models)==1 else np.array([]))


def plot_confusion_matrix(y_true, y_pred, labels) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title="Confusion Matrix")
    return fig


def model_feature_importance(pipe: Pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    # intentar nombres tras one-hot
    try:
        pre: ColumnTransformer = pipe.named_steps["prep"]
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        all_names = list(num_cols) + ohe_names
    except Exception:
        all_names = feature_names

    mdl = pipe.named_steps["model"]
    importances = None
    if hasattr(mdl, "feature_importances_"):
        importances = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        coefs = mdl.coef_
        importances = np.abs(coefs if coefs.ndim == 1 else coefs.mean(axis=0))

    if importances is None:
        return None

    imp = pd.DataFrame({"feature": all_names[:len(importances)], "importance": importances})
    imp = imp.sort_values("importance", ascending=False).head(30)
    return imp

# ------------------------
# App Body
# ------------------------

st.title("Lab 10 — Interactive Dashboard (Streamlit)")
st.markdown(
    """
This dashboard is designed for decision-makers:
- Explore the dataset interactively
- Apply **global filters** and see multiple views update
- Train and compare **3 simple models** (classification or regression)
- Drill down with **linked visuals** and **what-if controls**
    """
)

# --- Load & infer types
with st.spinner("Loading data…"):
    df = load_data("test.csv")
if df.empty:
    st.stop()

num_cols, cat_cols, dt_cols = split_cols(df)
if dt_cols:
    df = coerce_datetimes(df, dt_cols)

# --- Sidebar: display & chart choices
st.sidebar.markdown("### Display Options")
color_col = st.sidebar.selectbox("Color by (optional)", options=[None] + cat_cols + num_cols, index=0)
size_col = st.sidebar.selectbox("Size by (optional - numeric)", options=[None] + num_cols, index=0)

st.sidebar.markdown("### Chart Controls")
hist_bins = st.sidebar.slider("Histogram bins", min_value=5, max_value=100, value=30)
point_opacity = st.sidebar.slider("Scatter/Points opacity", min_value=0.1, max_value=1.0, value=0.8)
trendline_opt = st.sidebar.selectbox("Scatter trendline", options=[None, "ols", "lowess"], index=0)
bar_topn = st.sidebar.slider("Bar: Top N categories", 5, 50, 20)
bar_normalize = st.sidebar.checkbox("Bar: Normalize to %", value=False)
cor_method = st.sidebar.selectbox("Correlation method", options=["pearson", "kendall", "spearman"], index=0)
scattermatrix_dims = st.sidebar.slider("Scatter Matrix dimensions (first N numeric)", 3, 8, min(5, max(3, len(num_cols))))

st.sidebar.markdown("### Global What-If Controls")
sample_frac = st.sidebar.slider("Sample fraction", 0.1, 1.0, 1.0, 0.05)
outlier_clip = st.sidebar.checkbox("Clip numeric outliers (percentiles)", value=False)
low_p = st.sidebar.slider("Lower percentile", 0.0, 10.0, 1.0, 0.5)
high_p = st.sidebar.slider("Upper percentile", 90.0, 100.0, 99.0, 0.5)

whatif_col = st.sidebar.selectbox("What-if column (numeric)", options=[None] + num_cols, index=0)
whatif_flag_name = "__whatif_flag__"
whatif_thresh = None
use_flag_as_color = False
filter_high_only = False
if whatif_col:
    cmin, cmax = float(df[whatif_col].min()), float(df[whatif_col].max())
    whatif_thresh = st.sidebar.slider(f"Threshold for {whatif_col}", cmin, cmax, (cmin + cmax)/2)
    use_flag_as_color = st.sidebar.checkbox("Color by what-if flag", value=False)
    filter_high_only = st.sidebar.checkbox("Filter: keep only ≥ threshold", value=False)

# --- Global filters (linked)
filtered = apply_global_filters(df, num_cols, cat_cols, dt_cols)

# Sampling
if sample_frac < 1.0 and len(filtered) > 0:
    filtered = filtered.sample(frac=sample_frac, random_state=42)

# What-if flag
if whatif_col and whatif_thresh is not None and whatif_col in filtered.columns:
    filtered[whatif_flag_name] = (filtered[whatif_col] >= whatif_thresh).astype(int)
    if filter_high_only:
        filtered = filtered[filtered[whatif_flag_name] == 1]

# Clip outliers
if outlier_clip and len(filtered) > 0 and num_cols:
    lp = low_p/100.0
    hp = high_p/100.0
    for c in num_cols:
        lo = filtered[c].quantile(lp)
        hi = filtered[c].quantile(hp)
        filtered[c] = filtered[c].clip(lo, hi)

effective_color = whatif_flag_name if use_flag_as_color and (whatif_col and whatif_thresh is not None) else color_col

st.markdown("#### Data Preview (filtered)")
st.dataframe(filtered.head(200))

# ------------------------
# 1) Exploratory (>= 8 visuals)
# ------------------------
expander = st.expander("Exploratory Visuals", expanded=True)
with expander:
    c1, c2, c3 = st.columns(3)

    # 1. Histogram
    with c1:
        col_hist = st.selectbox("Histogram column", options=num_cols or list(df.columns), index=0)
        if col_hist:
            log_y = st.checkbox("Log scale (Y)", value=False, key=f"hist_log_{col_hist}")
            fig = px.histogram(filtered, x=col_hist, nbins=hist_bins, color=effective_color)
            fig.update_yaxes(type='log' if log_y else 'linear')
            st.plotly_chart(fig, use_container_width=True)

    # 2. Box plot
    with c2:
        if num_cols and cat_cols:
            y_col = st.selectbox("Box Y (numeric)", options=num_cols, index=0, key="boxy")
            x_col = st.selectbox("Box X (category)", options=cat_cols, index=0, key="boxx")
            pts = st.checkbox("Show points", value=False, key="boxpts")
            fig = px.box(filtered, x=x_col, y=y_col, color=effective_color, points="all" if pts else False)
            st.plotly_chart(fig, use_container_width=True)

    # 3. Scatter
    with c3:
        if len(num_cols) >= 2:
            x_sc = st.selectbox("Scatter X", options=num_cols, index=0, key="scx")
            y_sc = st.selectbox("Scatter Y", options=num_cols, index=1 if len(num_cols) > 1 else 0, key="scy")
            add_ref = st.checkbox("Add y=x ref line", value=False, key=f"sc_ref_{x_sc}_{y_sc}")
            fig = px.scatter(filtered, x=x_sc, y=y_sc, color=effective_color, size=size_col, opacity=point_opacity, trendline=trendline_opt)
            if add_ref and pd.api.types.is_numeric_dtype(filtered[x_sc]) and pd.api.types.is_numeric_dtype(filtered[y_sc]):
                try:
                    lo = float(min(filtered[x_sc].min(), filtered[y_sc].min()))
                    hi = float(max(filtered[x_sc].max(), filtered[y_sc].max()))
                    fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash='dash'))
                except Exception:
                    pass
            st.plotly_chart(fig, use_container_width=True)

    c4, c5, c6 = st.columns(3)

    # 4. Bar chart (top categories)
    with c4:
        if cat_cols:
            bar_col = st.selectbox("Bar — categorical", options=cat_cols, index=0, key="barcat")
            top_counts = filtered[bar_col].value_counts().head(bar_topn).reset_index()
            top_counts.columns = [bar_col, "count"]
            if bar_normalize and len(filtered) > 0:
                top_counts["count"] = (top_counts["count"] / len(filtered) * 100.0)
                ylab = "percent"
            else:
                ylab = "count"
            horizontal = st.checkbox("Horizontal bars", value=True, key=f"bar_h_{bar_col}")
            if horizontal:
                fig = px.bar(top_counts, x="count", y=bar_col, orientation='h', labels={"count": ylab}, color=effective_color)
            else:
                fig = px.bar(top_counts, x=bar_col, y="count", labels={"count": ylab}, color=effective_color)
            st.plotly_chart(fig, use_container_width=True)

    # 5. Correlation heatmap
    with c5:
        if len(num_cols) >= 2:
            with pd.option_context('mode.use_inf_as_na', True):
                corr = filtered[num_cols].corr(method=cor_method, numeric_only=True)
            fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu", origin="lower")
            fig.update_layout(title=f"Correlation ({cor_method})")
            st.plotly_chart(fig, use_container_width=True)

    # 6. Scatter Matrix
    with c6:
        if len(num_cols) >= 3:
            dims = num_cols[:scattermatrix_dims]
            fig = px.scatter_matrix(filtered, dimensions=dims, color=effective_color, opacity=point_opacity)
            st.plotly_chart(fig, use_container_width=True)

    # 7. Time series (if datetime)
    if dt_cols:
        st.markdown("##### Time Series")
        ts_col = st.selectbox("Y (numeric)", options=num_cols, index=0, key="tsy")
        dtc = dt_cols[0]
        freq = st.selectbox("Resample freq", options=["None", "D", "W", "M", "Q"], index=0)
        agg = st.selectbox("Aggregation", options=["sum", "mean", "median"], index=1)
        ts = filtered[[dtc, ts_col]].dropna().sort_values(dtc)
        if freq != "None":
            ts = ts.set_index(dtc).resample(freq).agg({ts_col: agg}).reset_index()
        fig = px.line(ts, x=dtc, y=ts_col)
        st.plotly_chart(fig, use_container_width=True)

    # 8. Map (if lat/lon present)
    lat_candidates = [c for c in filtered.columns if c.lower() in ("lat", "latitude")]
    lon_candidates = [c for c in filtered.columns if c.lower() in ("lon", "lng", "longitude")]
    if lat_candidates and lon_candidates:
        lat_c, lon_c = lat_candidates[0], lon_candidates[0]
        st.markdown("##### Map")
        map_point_size = st.slider("Map point size", 2, 20, 8)
        dfm = filtered.dropna(subset=[lat_c, lon_c]).head(3000)
        fig = px.scatter_mapbox(dfm, lat=lat_c, lon=lon_c, color=color_col, zoom=3, height=400, size_max=map_point_size)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------
# 2) Modeling (3 models) + linked outputs with hyperparameters
# ------------------------
st.markdown("---")
st.header("Modeling — Compare up to 3 Models")

# Sidebar hyperparameters
st.sidebar.markdown("### Model Hyperparameters")
cv_test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
rf_n_estimators = st.sidebar.slider("RandomForest: n_estimators", 50, 500, 200, 50)
rf_max_depth = st.sidebar.selectbox("RandomForest: max_depth", options=[None, 5, 10, 20, 50], index=0)
knn_k = st.sidebar.slider("KNN: k neighbors", 3, 31, 7, 2)

if len(df.columns) >= 2:
    target, task = pick_target(filtered, num_cols, cat_cols)
    if target:
        metrics_df, fitted, metrics, youts = train_and_eval(
            filtered, target, task, cv_test_size, rf_n_estimators, rf_max_depth, knn_k
        )

        st.subheader("Model Performance Table")
        st.dataframe(metrics_df.style.format({k: "{:.3f}" for k in metrics_df.columns}))

        chosen_for_viz = st.multiselect(
            "Select models to visualize (max 2 for plots)",
            options=list(fitted.keys()),
            default=list(fitted.keys())[:2]
        )

        v1, v2 = st.columns(2)

        if task == "classification":
            with v1:
                st.markdown("**Confusion Matrix**")
                if len(chosen_for_viz) >= 1:
                    for name in chosen_for_viz[:2]:
                        pipe = fitted[name]
                        X = filtered.drop(columns=[target])
                        y = filtered[target]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=cv_test_size, random_state=42, stratify=y
                        )
                        y_pred = pipe.predict(X_test)
                        labels = sorted(y_test.unique())
                        fig = plot_confusion_matrix(y_test, y_pred, labels)
                        fig.update_layout(title=f"Confusion Matrix — {name}")
                        st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.markdown("**ROC Curves (if supported)**")
                for name in chosen_for_viz[:2]:
                    pipe = fitted[name]
                    X = filtered.drop(columns=[target])
                    y = filtered[target]
                    if y.nunique() == 2:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=cv_test_size, random_state=42, stratify=y
                        )
                        if hasattr(pipe.named_steps["model"], "predict_proba"):
                            proba = pipe.predict_proba(X_test)[:, 1]
                            pos_label = list(sorted(y.unique()))[1]
                            fpr, tpr, _ = roc_curve(y_test, proba, pos_label=pos_label)
                            roc_auc = auc(fpr, tpr)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} AUC={roc_auc:.3f}'))
                            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                            fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='ROC Curve')
                            st.plotly_chart(fig, use_container_width=True)

        else:  # regression
            with v1:
                st.markdown("**Residuals Plot**")
                if chosen_for_viz:
                    name = chosen_for_viz[0]
                    pipe = fitted[name]
                    X = filtered.drop(columns=[target])
                    y = filtered[target]
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cv_test_size, random_state=42)
                    y_pred = pipe.predict(X_te)
                    resid = y_te - y_pred
                    fig = px.scatter(x=y_pred, y=resid, labels={"x":"Predicted", "y":"Residuals"}, opacity=point_opacity)
                    fig.add_hline(y=0, line_dash="dash")
                    fig.update_layout(title=f"Residuals — {name}")
                    st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.markdown("**Predicted vs Actual**")
                if chosen_for_viz:
                    name = chosen_for_viz[0]
                    pipe = fitted[name]
                    X = filtered.drop(columns=[target])
                    y = filtered[target]
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cv_test_size, random_state=42)
                    y_pred = pipe.predict(X_te)
                    fig = px.scatter(x=y_te, y=y_pred, labels={"x":"Actual", "y":"Predicted"}, opacity=point_opacity)
                    fig.add_shape(type='line', x0=y_te.min(), y0=y_te.min(), x1=y_te.max(), y1=y_te.max(), line=dict(dash='dash'))
                    fig.update_layout(title=f"Actual vs Predicted — {name}")
                    st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Features (Importance / |coef|)")
        for name in chosen_for_viz[:2]:
            pipe = fitted[name]
            imp_df = model_feature_importance(pipe, feature_names=[c for c in filtered.columns if c != target])
            if imp_df is not None and not imp_df.empty:
                fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
                fig.update_layout(title=f"Feature Importance — {name}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{name}: feature importances/coefficients not available.")

# ------------------------
# 2.b) Extra What-If Visuals (2 dinámicas nuevas)
# ------------------------
if 'fitted' in locals() and isinstance(fitted, dict) and len(fitted) > 0 and 'metrics_df' in locals() and 'target' in locals() and target and task == 'classification':
    st.markdown("---")
    st.header("What-If Experiments (Interactive)")

    # --- What-If 1: impacto de filtro por palabras en accuracy
    st.subheader("What-If #1 — Accuracy impact by text filter")
    model_for_whatif = st.selectbox("Model to test", options=list(fitted.keys()), key="whatif_model")
    text_like_cols = [c for c in filtered.columns if filtered[c].dtype == object]
    col_to_search = st.selectbox("Column to search (substring)", options=text_like_cols or [None], index=0)
    kw_input = st.text_input("Keywords (comma-separated)", value="fire, flood, quake")
    mode = st.radio("Mode", options=["Include rows containing ANY", "Exclude rows containing ANY"], horizontal=True)

    def contains_any(text: str, kws: list[str]) -> bool:
        t = str(text).lower()
        return any(k.strip().lower() in t for k in kws if k.strip())

    df_base = filtered.copy()
    baseline_name = model_for_whatif

    from sklearn.pipeline import Pipeline as SkPipeline

    def make_pipe(name: str, task_: str):
        numc = [c for c in df_base.columns if c != target and pd.api.types.is_numeric_dtype(df_base[c])]
        catc = [c for c in df_base.columns if c != target and c not in numc]
        pre = ColumnTransformer([
            ("num", SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), numc),
            ("cat", SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), catc)
        ])
        mdl_map = build_models(task_, rf_n_estimators, rf_max_depth, knn_k)
        return SkPipeline(steps=[("prep", pre), ("model", mdl_map[name])])

    Xb = df_base.drop(columns=[target])
    yb = df_base[target]
    strat = yb if yb.nunique() > 1 else None
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=cv_test_size, random_state=42, stratify=strat)
    pipe_base = make_pipe(baseline_name, task)
    pipe_base.fit(Xb_tr, yb_tr)
    yb_pred = pipe_base.predict(Xb_te)
    base_acc = accuracy_score(yb_te, yb_pred)

    kws = [k for k in (kw_input.split(",") if kw_input else [])]
    if col_to_search and len(kws) > 0:
        mask_any = df_base[col_to_search].apply(lambda t: contains_any(t, kws))
        df_kw = df_base[mask_any] if mode.startswith("Include") else df_base[~mask_any]
    else:
        df_kw = df_base.copy()

    if len(df_kw) >= 20 and df_kw[target].nunique() >= 2:
        Xk = df_kw.drop(columns=[target])
        yk = df_kw[target]
        stratk = yk if yk.nunique() > 1 else None
        Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(Xk, yk, test_size=cv_test_size, random_state=42, stratify=stratk)
        pipe_kw = make_pipe(baseline_name, task)
        pipe_kw.fit(Xk_tr, yk_tr)
        yk_pred = pipe_kw.predict(Xk_te)
        kw_acc = accuracy_score(yk_te, yk_pred)

        cmp_df = pd.DataFrame({
            "scenario": ["Baseline (current filters)", f"What-If ({mode.split()[0]} ANY: {len(kws)} kw)"],
            "accuracy": [base_acc, kw_acc]
        })
        fig = px.bar(cmp_df, x="scenario", y="accuracy", text="accuracy")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".2f", title="Accuracy impact of keyword filter")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Rows used — Baseline: {len(df_base)}, What-If: {len(df_kw)}")
    else:
        st.info("Not enough rows or only one class after the filter — adjust your keywords or mode.")

    # --- What-If 2: threshold tuner (binario)
    st.subheader("What-If #2 — Threshold tuner (Precision/Recall/F1/Accuracy)")
    model_for_thresh = st.selectbox("Model for threshold tuning", options=list(fitted.keys()), key="thresh_model")

    df_cls = df_base.copy()
    Xc = df_cls.drop(columns=[target])
    yc = df_cls[target]
    if yc.nunique() == 2:
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=cv_test_size, random_state=42, stratify=yc)
        pipe_cls = make_pipe(model_for_thresh, task)
        pipe_cls.fit(Xc_tr, yc_tr)

        proba = None
        if hasattr(pipe_cls.named_steps["model"], "predict_proba"):
            proba = pipe_cls.predict_proba(Xc_te)[:, 1]
        elif hasattr(pipe_cls.named_steps["model"], "decision_function"):
            scores = pipe_cls.decision_function(Xc_te)
            smin, smax = float(scores.min()), float(scores.max())
            proba = (scores - smin) / (smax - smin + 1e-9)

        if proba is not None:
            ths = np.linspace(0.05, 0.95, 19)
            rows = []
            pos_label = sorted(yc.unique())[1]
            for t in ths:
                yp = (proba >= t).astype(int)
                labels_sorted = sorted(yc.unique())
                yp_lbl = np.where(yp == 1, labels_sorted[1], labels_sorted[0])
                rows.append({
                    "threshold": t,
                    "accuracy": accuracy_score(yc_te, yp_lbl),
                    "precision": precision_score(yc_te, yp_lbl, average="binary", pos_label=pos_label, zero_division=0),
                    "recall": recall_score(yc_te, yp_lbl, average="binary", pos_label=pos_label, zero_division=0),
                    "f1": f1_score(yc_te, yp_lbl, average="binary", pos_label=pos_label, zero_division=0)
                })
            perf = pd.DataFrame(rows)
            perf_fig = px.line(perf, x="threshold", y=["accuracy", "precision", "recall", "f1"], markers=True)
            perf_fig.update_layout(title="Metrics vs Threshold", yaxis_tickformat=".2f")
            st.plotly_chart(perf_fig, use_container_width=True)

            sel_t = st.slider("Select threshold", 0.0, 1.0, 0.5, 0.01)
            yp = (proba >= sel_t).astype(int)
            labels_sorted = sorted(yc.unique())
            yp_lbl = np.where(yp == 1, labels_sorted[1], labels_sorted[0])
            cm = confusion_matrix(yc_te, yp_lbl, labels=labels_sorted)
            cm_fig = px.imshow(cm, text_auto=True, x=labels_sorted, y=labels_sorted, labels=dict(x="Pred", y="Real", color="Count"))
            cm_fig.update_layout(title=f"Confusion Matrix @ threshold={sel_t:.2f}")
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.info("Selected model does not expose probabilities/scores for threshold tuning.")
    else:
        st.info("Threshold tuner applies to binary classification only.")

# ------------------------
# 3) Model comparison table
# ------------------------
st.markdown("---")
st.header("Compare Models")
if 'metrics_df' in locals():
    to_compare = st.multiselect("Select 2 or 3 models to compare", options=list(metrics_df.index), default=list(metrics_df.index)[:2])
    if to_compare:
        st.dataframe(metrics_df.loc[to_compare].style.format({k: "{:.3f}" for k in metrics_df.columns}))

# ------------------------
# 4) Executive Notes
# ------------------------
st.markdown("---")
st.subheader("Executive Notes")
notes = st.text_area(
    "Key patterns, drivers, and decisions (editable):",
    height=120,
    value=(
        "• Use the filters to segment and watch KPIs shift across all visuals.\n"
        "• Compare 3 models in the performance table; dive deeper with confusion matrices or residuals.\n"
        "• Inspect top features to understand drivers before acting.\n"
    ),
)
st.caption("These notes are stored only in session state; copy/export as needed.")
