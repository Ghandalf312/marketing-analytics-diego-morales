import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]
BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "uplift_campaign.csv")


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(
    train_t[FEATURES],
    train_t["responded_60d"]
    )

    modelo_control = LogisticRegression(max_iter=1000).fit(
        train_c[FEATURES],
        train_c["responded_60d"]
    )

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)

def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(
        np.where(es_tratado, respondio, 0)
    )
    acumulado_control = np.cumsum(
        np.where(~es_tratado, respondio, 0)
    )

    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)

    xs = [0.0]
    ys = [0.0]

    for i in np.linspace(1, n, steps).astype(int):
        nt = n_tratados_acum[i - 1]
        nc = n_control_acum[i - 1]

        tasa_t = (
            acumulado_tratados[i - 1] / nt
            if nt > 0 else 0
        )

        tasa_c = (
            acumulado_control[i - 1] / nc
            if nc > 0 else 0
        )

        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)

    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)

    area_azar = np.trapezoid(
        np.linspace(0, ys[-1], len(xs)),
        xs
    )

    return (area_modelo - area_azar) / n

# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption(
    "Esta aplicación permite identificar qué porcentaje de clientes conviene contactar "
    "para maximizar el valor incremental de una campaña."
)

df = load_data()

with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
    "Porcentaje de clientes a contactar",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- Curva Qini del T-learner ---

xs_uplift, ys_uplift = curva_qini(
    test,
    "uplift_estimado"
)

qini_uplift = coeficiente_qini(
    xs_uplift,
    ys_uplift,
    len(test)
)

st.subheader("Curva Qini del T-Learner")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    xs_uplift,
    ys_uplift,
    color=GREEN,
    linewidth=3,
    label=f"T-Learner (Qini={qini_uplift:+.3f})"
)

ax.plot(
    [0, 1],
    [0, ys_uplift[-1]],
    color=GRAY,
    linestyle="--",
    label="Al azar"
)

ax.set_xlabel("% de la base contactada")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Curva Qini")
ax.legend()

plt.tight_layout()

st.pyplot(fig)


# --- Política para el porcentaje seleccionado ---

n_contactar = int(
    len(test) * pct_contactar / 100
)

idx_seleccionados = (
    test
    .sort_values(
        "uplift_estimado",
        ascending=False
    )
    .head(n_contactar)
    .index
)

valor_cliente, n = valor_de_la_politica(
    test,
    idx_seleccionados
)

valor_total = valor_cliente * n

st.subheader("Política de contacto")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Clientes contactados",
    f"{n:,}"
)

col2.metric(
    "Valor incremental por cliente",
    f"${valor_cliente:,.2f}"
)

col3.metric(
    "Valor incremental total",
    f"${valor_total:,.2f}"
)

tabla_politica = pd.DataFrame({
    "% contactado": [pct_contactar],
    "Clientes contactados": [n],
    "Valor por cliente": [round(valor_cliente, 2)],
    "Valor total": [round(valor_total, 2)]
})

st.dataframe(
    tabla_politica,
    use_container_width=True
)

