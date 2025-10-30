# app.py — Interactive NLP Dashboard for Lab 10 (Disaster Tweets)
# ---------------------------------------------------------------
# Cómo ejecutar localmente:
#   1) Crea un entorno e instala dependencias:
#        pip install -r requirements.txt
#   2) Ejecuta:
#        streamlit run app.py
#
# requirements.txt (mínimo sugerido):
#   streamlit>=1.36
#   pandas>=2.2
#   numpy>=1.26
#   scikit-learn>=1.4
#   plotly>=5.22
#   wordcloud>=1.9
#   pillow>=10.3

from typing import List, Tuple, Dict, Optional
import re, math, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Lab 10 — NLP Dashboard", layout="wide")

# ------------------------
# Helpers
# ------------------------

def load_csv(name: str) -> Optional[pd.DataFrame]:
    """Load CSV if available in working dir."""
    try:
        if os.path.exists(name):
            return pd.read_csv(name)
    except Exception:
        pass
    return None


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    st.sidebar.markdown("### Carga de archivos")
    up_train = st.sidebar.file_uploader("Subir train.csv (opcional)", type=["csv"], key="train")
    up_test  = st.sidebar.file_uploader("Subir test.csv (opcional)", type=["csv"], key="test")

    train_df = pd.read_csv(up_train) if up_train is not None else load_csv("train.csv")
    test_df  = pd.read_csv(up_test)  if up_test  is not None else load_csv("test.csv")
    return train_df, test_df


def clean_text(s: str) -> str:
    s = re.sub(r"http\S+", " ", str(s))
    s = re.sub(r"[#@]\w+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" in df.columns:
        df["text_len"] = df["text"].astype(str).str.len()
        df["text_words"] = df["text"].astype(str).str.split().apply(len)
        df["text_clean"] = df["text"].astype(str).apply(clean_text)
        df["text_clean_words"] = df["text_clean"].str.split().apply(len)
    return df


def top_n_terms(corpus: List[str], n=20, ngram_range=(1,1)) -> pd.DataFrame:
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=2)
    X = tfidf.fit_transform(corpus)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(tfidf.get_feature_names_out())
    idx = np.argsort(-sums)[:n]
    return pd.DataFrame({"term": terms[idx], "score": sums[idx]})


def plot_wordcloud(texts: List[str]):
    wc = WordCloud(width=900, height=400, background_color="white").generate(" ".join(texts))
    return wc.to_image()

# ------------------------
# Load
# ------------------------
train_df, test_df = load_data()

if test_df is None and train_df is None:
    st.warning("Sube al menos test.csv para continuar.")
    st.stop()

st.title("Lab 10 — Interactive NLP Dashboard (Tweets de desastre)")

with st.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("train.csv", f"{train_df.shape if train_df is not None else '—'}")
    c2.metric("test.csv", f"{test_df.shape if test_df is not None else '—'}")
    c3.metric("Target disponible", "Sí" if train_df is not None and "target" in train_df.columns else "No")

# Usar test_df como base de exploración, y train_df para modelado si existe
base = test_df if test_df is not None else train_df
base = add_derived(base)

# ------------------------
# Global filters (linked)
# ------------------------
st.sidebar.markdown("### Filtros globales (enlazados)")

# Filtro por keyword
if "keyword" in base.columns:
    kws = sorted([k for k in base["keyword"].dropna().unique()])
    sel_k = st.sidebar.multiselect("keyword", options=kws, default=[])
else:
    sel_k = []

# Filtro por longitud de texto
len_min, len_max = int(base["text_len"].min() if "text_len" in base else 0), int(base["text_len"].max() if "text_len" in base else 100)
len_rng = st.sidebar.slider("Longitud de texto (caracteres)", min_value=len_min, max_value=len_max, value=(len_min, len_max))

# Búsqueda simple
query = st.sidebar.text_input("Buscar texto contiene… (min 3 chars)", "")

filtered = base.copy()
if sel_k:
    filtered = filtered[filtered["keyword"].isin(sel_k)]
if "text_len" in filtered.columns:
    filtered = filtered[(filtered["text_len"] >= len_rng[0]) & (filtered["text_len"] <= len_rng[1])]
if query and len(query) >= 3 and "text" in filtered.columns:
    filtered = filtered[filtered["text"].str.contains(query, case=False, na=False)]

st.markdown("#### Vista previa de datos (filtrados)")
st.dataframe(filtered.head(200))

# ------------------------
# 1) Exploratorio (≥ 6 visualizaciones interactivas)
# ------------------------
exp = st.expander("Exploración interactiva", expanded=True)
with exp:
    a, b, c = st.columns(3)
    # 1. Distribución de palabras
    with a:
        if "text_words" in filtered.columns:
            fig = px.histogram(filtered, x="text_words", nbins=40)
            fig.update_layout(title="Distribución de tamaño de texto (palabras)")
            st.plotly_chart(fig, use_container_width=True)
    # 2. Top keywords
    with b:
        if "keyword" in filtered.columns:
            top_kw = filtered["keyword"].value_counts().head(20).reset_index()
            top_kw.columns = ["keyword", "count"]
            fig = px.bar(top_kw, x="keyword", y="count")
            fig.update_layout(title="Top keywords")
            st.plotly_chart(fig, use_container_width=True)
    # 3. Terms (unigrams)
    with c:
        if "text_clean" in filtered.columns and not filtered.empty:
            terms = top_n_terms(filtered["text_clean"].tolist(), n=20, ngram_range=(1,1))
            fig = px.bar(terms, x="score", y="term", orientation="h")
            fig.update_layout(title="Top términos (unigramas)")
            st.plotly_chart(fig, use_container_width=True)

    d, e, f = st.columns(3)
    # 4. Bigrams
    with d:
        if "text_clean" in filtered.columns and not filtered.empty:
            bigr = top_n_terms(filtered["text_clean"].tolist(), n=20, ngram_range=(2,2))
            fig = px.bar(bigr, x="score", y="term", orientation="h")
            fig.update_layout(title="Top términos (bigramas)")
            st.plotly_chart(fig, use_container_width=True)
    # 5. Wordcloud
    with e:
        if "text_clean" in filtered.columns and not filtered.empty:
            img = plot_wordcloud(filtered["text_clean"].tolist()[:5000])
            st.image(img, caption="WordCloud (muestra)", use_column_width=True)
    # 6. Tabla de ejemplos
    with f:
        cols = [c for c in ["id","keyword","location","text"] if c in filtered.columns]
        st.dataframe(filtered[cols].head(20))

# ------------------------
# 2) Modelado (3 modelos) con TF-IDF (si hay train.csv con 'target')
# ------------------------
st.markdown("---")
st.header("Modelado — Comparación de 3 clasificadores (TF-IDF)")

if train_df is not None and "target" in train_df.columns and "text" in train_df.columns:
    train_df = add_derived(train_df)
    # Preparación
    X = train_df["text_clean"]
    y = train_df["target"]

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Modelos
    models: Dict[str, Pipeline] = {
        "LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("svm", CalibratedClassifierCV(LinearSVC(), method="sigmoid"))
        ]),
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("nb", MultinomialNB())
        ]),
    }

    pick = st.multiselect("Elige modelos (máx 3)", list(models.keys()), default=list(models.keys()))

    results = {}
    fitted = {}
    for name in pick:
        pipe = models[name]
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        y_pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

    if results:
        metrics_df = pd.DataFrame(results).T
        st.subheader("Tabla de desempeño")
        st.dataframe(metrics_df.style.format({c: "{:.3f}" for c in metrics_df.columns}))

        # Enlaces: seleccionar un modelo para ver CM/ROC/PR y top features
        chosen = st.selectbox("Modelo para diagnósticos", options=list(fitted.keys()))
        pipe = fitted[chosen]
        y_pred = pipe.predict(X_test)

        g1, g2 = st.columns(2)
        with g1:
            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig = px.imshow(cm, text_auto=True, x=labels, y=labels, labels=dict(x="Predicho", y="Real", color="Cuenta"))
            fig.update_layout(title=f"Matriz de confusión — {chosen}")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            # ROC/PR si binario
            if len(np.unique(y)) == 2:
                # necesitamos probabilidades
                proba = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)[:,1]
                    except Exception:
                        proba = None
                if proba is None and hasattr(pipe, "decision_function"):
                    try:
                        proba = pipe.decision_function(X_test)
                    except Exception:
                        proba = None
                if proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, proba, pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.3f}'))
                    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                    fig.update_layout(title=f"ROC — {chosen}", xaxis_title='FPR', yaxis_title='TPR')
                    st.plotly_chart(fig, use_container_width=True)

                    from sklearn.metrics import precision_recall_curve
                    prec, rec, _ = precision_recall_curve(y_test, proba, pos_label=1)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name='PR'))
                    fig2.update_layout(title=f"Precision-Recall — {chosen}", xaxis_title='Recall', yaxis_title='Precision')
                    st.plotly_chart(fig2, use_container_width=True)

        # Importancia de términos (coeficientes)
        st.subheader("Términos más influyentes (solo modelos lineales)")
        if chosen in ("LogReg", "LinearSVM"):
            # Recuperar el vectorizador y el clasificador
            vec = pipe.named_steps["tfidf"]
            clf_key = "clf" if "clf" in pipe.named_steps else "svm"
            clf = pipe.named_steps[clf_key]
            feats = vec.get_feature_names_out()
            # Para CalibratedClassifierCV (LinearSVC calibrado), no hay coef_ directa
            if hasattr(clf, "coef_"):
                coefs = clf.coef_.ravel()
                top_pos_idx = np.argsort(-coefs)[:20]
                top_neg_idx = np.argsort(coefs)[:20]
                pos_df = pd.DataFrame({"term": feats[top_pos_idx], "coef": coefs[top_pos_idx]})
                neg_df = pd.DataFrame({"term": feats[top_neg_idx], "coef": coefs[top_neg_idx]})
                h1, h2 = st.columns(2)
                with h1:
                    st.plotly_chart(px.bar(pos_df, x="coef", y="term", orientation="h", title="Señales de clase positiva"), use_container_width=True)
                with h2:
                    st.plotly_chart(px.bar(neg_df, x="coef", y="term", orientation="h", title="Señales de clase negativa"), use_container_width=True)
            else:
                st.info("Este modelo no expone coeficientes; se omite el ranking de términos.")

    # Predicción sobre test.csv si existe
    if test_df is not None and "text" in test_df.columns and results:
        with st.spinner("Generando predicciones sobre test.csv…"):
            st.subheader("Inferencia sobre test.csv (con el modelo seleccionado arriba)")
            tdf = add_derived(test_df)
            preds = pipe.predict(tdf["text_clean"])
            out = test_df.copy()
            out["pred"] = preds
            st.dataframe(out.head(30))
            st.download_button(
                "Descargar predicciones CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predicciones.csv",
                mime="text/csv"
            )

else:
    st.info("Para entrenar y comparar modelos, sube train.csv con la columna 'target'. Aún así, puedes explorar test.csv arriba.")

# ------------------------
# 3) Selector de visualizaciones y notas ejecutivas
# ------------------------
st.markdown("---")
st.header("Panel personalizable y notas")

viz_opts = {
    "Distribución de palabras": "words",
    "Top keywords": "kw",
    "Unigramas": "uni",
    "Bigramas": "bi",
    "WordCloud": "wc",
    "Tabla de ejemplos": "tbl"
}
select_viz = st.multiselect("Elige qué gráficas mostrar", options=list(viz_opts.keys()), default=list(viz_opts.keys()))

# Re-render rápido según selección (reusa 'filtered')
cols = st.columns(3)
slot_map = {0: [], 1: [], 2: []}
for i, name in enumerate(select_viz):
    slot_map[i % 3].append(name)

for col_idx, items in slot_map.items():
    with cols[col_idx]:
        for name in items:
            code = viz_opts[name]
            if code == "words" and "text_words" in filtered.columns:
                st.plotly_chart(px.histogram(filtered, x="text_words", nbins=40, title="Distribución de tamaño de texto"), use_container_width=True)
            elif code == "kw" and "keyword" in filtered.columns:
                top_kw = filtered["keyword"].value_counts().head(20).reset_index()
                top_kw.columns = ["keyword","count"]
                st.plotly_chart(px.bar(top_kw, x="keyword", y="count", title="Top keywords"), use_container_width=True)
            elif code == "uni" and "text_clean" in filtered.columns and not filtered.empty:
                terms = top_n_terms(filtered["text_clean"].tolist(), n=20, ngram_range=(1,1))
                st.plotly_chart(px.bar(terms, x="score", y="term", orientation="h", title="Unigramas"), use_container_width=True)
            elif code == "bi" and "text_clean" in filtered.columns and not filtered.empty:
                bigr = top_n_terms(filtered["text_clean"].tolist(), n=20, ngram_range=(2,2))
                st.plotly_chart(px.bar(bigr, x="score", y="term", orientation="h", title="Bigramas"), use_container_width=True)
            elif code == "wc" and "text_clean" in filtered.columns and not filtered.empty:
                st.image(plot_wordcloud(filtered["text_clean"].tolist()[:5000]), caption="WordCloud", use_column_width=True)
            elif code == "tbl":
                cols_show = [c for c in ["id","keyword","location","text"] if c in filtered.columns]
                st.dataframe(filtered[cols_show].head(20))

st.subheader("Notas ejecutivas")
st.text_area("Puntos clave para toma de decisiones:", height=120, value=(
    "• Use los filtros (keyword, longitud, búsqueda) para segmentar las vistas.\n"
    "• Compare 3 modelos (si hay train.csv) y valide con ROC/PR y matriz de confusión.\n"
    "• Revise términos influyentes para entender drivers de clasificación.\n"
))
