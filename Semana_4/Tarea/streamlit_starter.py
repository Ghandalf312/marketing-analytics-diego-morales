"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def cargar():
    catalogo = pd.read_csv(BASE_DIR / "data" / "catalogo.csv")
    hist = pd.read_csv(BASE_DIR / "data" / "ordenes_historial.csv")
    hold = pd.read_csv(BASE_DIR / "data" / "ordenes_holdout.csv")
    exp = pd.read_csv(BASE_DIR / "data" / "experimento.csv")
    clientes = pd.read_csv(BASE_DIR / "data" / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)   # (cosine_similarity entre las columnas de X)

# SVD con 8 factores
svd = TruncatedSVD(
    n_components=8,
    random_state=42
)

Z = svd.fit_transform(X)
S_svd = Z @ svd.components_

def puntajes(metodo, x_vec, K, fila=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it
    if metodo == "SVD (8 factores)":
        return S_svd[fila]
    # TODO 3: filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    if fila is not None:
        sims[fila] = 0

    vecinos = np.argsort(-sims)[:K]

    return sims[vecinos] @ X[vecinos]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption(
    "Esta aplicación permite explorar distintos métodos de recomendación y seleccionar productos relevantes para cada cliente."
)

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")

    metodo = st.selectbox(
        "Método de recomendación",
        [
            "Popularidad",
            "Filtrado por ítem",
            "Filtrado por cliente",
            "SVD (8 factores)"
        ],
        index=1
    )

    if metodo == "Filtrado por cliente":
        K = st.slider(
            "Número de vecinos",
            min_value=5,
            max_value=100,
            value=30,
            step=5
        )
    else:
        K = 30
    
    st.divider()

    with st.expander("Integrantes - Grupo 8"):
        st.markdown("""
    **Diego Javier Morales Monzón**  
    Carné: 1132119

    **Estuardo José Villeda Navarro**  
    Carné: 1003519

    **Jorge Waldemar García Baldizón**  
    Carné: 1220019

    **José Daniel De León Chang**  
    Carné: 1170419
            """)

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    st.write("Ya compró:", [productos[j] for j in np.where(x_vec > 0)[0]])

    # TODO 5: muestren las 3 recomendaciones con su puntaje y, para cada una, el "porque"
    #         - por ítem: "porque compró X (similitud 0.87)"
    #         - por cliente: "N de sus K vecinos más parecidos lo compraron"
    #         - popularidad: "es de lo más comprado"
    #         (y qué hacen con un cliente SIN historial: los métodos colaborativos dan puntaje 0)
    
        # Si no tiene historial, usamos Popularidad como respaldo
    metodo_usado = metodo

    if x_vec.sum() == 0 and metodo != "Popularidad":
        metodo_usado = "Popularidad"
        st.info(
            "Este cliente no tiene historial de compras. "
            "Se utilizará Popularidad como respaldo."
        )

    scores = puntajes(
        metodo_usado,
        x_vec,
        K,
        fila=fila
    )

    recomendaciones = top3(scores, x_vec)

    st.subheader("Top 3 recomendaciones")

    # Información adicional para explicar filtrado por cliente
    if metodo_usado == "Filtrado por cliente":
        sims_cliente = cosine_similarity(
            x_vec[None, :],
            X
        )[0]

        sims_cliente[fila] = 0
        vecinos = np.argsort(-sims_cliente)[:K]

    comprados = np.where(x_vec > 0)[0]

    for posicion, j in enumerate(recomendaciones, start=1):

        if metodo_usado == "Popularidad":
            porque = (
                "Es uno de los productos más comprados "
                "por los clientes."
            )

        elif metodo_usado == "Filtrado por ítem":
            if len(comprados) > 0:
                similitudes = S_it[comprados, j]
                mejor_pos = np.argmax(similitudes)
                producto_base = comprados[mejor_pos]
                similitud = similitudes[mejor_pos]

                porque = (
                    f"Porque compró {productos[producto_base]} "
                    f"(similitud {similitud:.2f})."
                )
            else:
                porque = "No existe suficiente historial."

        elif metodo_usado == "SVD (8 factores)":
            porque = (
                "El modelo detectó afinidad con este producto "
                "a partir de patrones latentes en su historial de compras."
            )

        else:
            n_vecinos = int(
                X[vecinos, j].sum()
            )

            porque = (
                f"{n_vecinos} de sus {K} vecinos "
                "más parecidos lo compraron."
            )

        st.markdown(
            f"### {posicion}. {productos[j]}"
        )

        st.write(
            f"**Puntaje:** {scores[j]:.2f}"
        )

        st.write(
            f"**¿Por qué?** {porque}"
        )

with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    st.subheader("Comparación de métodos")

    elegibles = NUEVO.sum(axis=1) > 0
    idx_control = np.where(
        es_control & elegibles
    )[0]

    n_evaluar = st.slider(
        "Número de clientes de control a evaluar",
        min_value=100,
        max_value=len(idx_control),
        value=min(500, len(idx_control)),
        step=100
    )

    # Muestra reproducible de clientes
    rng = np.random.default_rng(42)

    if n_evaluar < len(idx_control):
        idx_eval = rng.choice(
            idx_control,
            size=n_evaluar,
            replace=False
        )
    else:
        idx_eval = idx_control

    def evaluar(scores, indices):
        aciertos = []
        productos_recomendados = set()

        for pos, u in enumerate(indices):
            rec = top3(
                scores[pos],
                X[u]
            )

            aciertos.append(
                NUEVO[u, rec].sum() > 0
            )

            productos_recomendados.update(
                rec.tolist()
            )

        return (
            np.mean(aciertos),
            len(productos_recomendados)
        )

    with st.spinner("Calculando comparación..."):

        # Popularidad
        scores_pop = np.tile(
            popularidad.astype(float),
            (len(idx_eval), 1)
        )

        # Filtrado por ítem
        scores_item = X[idx_eval] @ S_it
        
        # Filtrado por SVD
        scores_svd = S_svd[idx_eval]

        # Filtrado por cliente
        sims_eval = cosine_similarity(
            X[idx_eval],
            X
        )

        sims_eval[
            np.arange(len(idx_eval)),
            idx_eval
        ] = 0

        vecinos_eval = np.argsort(
            -sims_eval,
            axis=1
        )[:, :K]

        scores_user = np.zeros(
            (len(idx_eval), len(productos)),
            dtype=float
        )

        for i in range(len(idx_eval)):
            vec = vecinos_eval[i]

            scores_user[i] = (
                sims_eval[i, vec]
                @ X[vec]
            )

        resultados = []

        for nombre, scores in {
            "Popularidad": scores_pop,
            "Filtrado por ítem": scores_item,
            "Filtrado por cliente": scores_user,
            "SVD (8 factores)": scores_svd
        }.items():

            hit, cobertura = evaluar(
                scores,
                idx_eval
            )

            resultados.append({
                "Método": nombre,
                "Hit rate@3": hit,
                "Cobertura": cobertura
            })

        tabla = pd.DataFrame(resultados)

    st.dataframe(
        tabla.style.format({
            "Hit rate@3": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Evaluación realizada sobre {len(idx_eval):,} "
        "clientes del grupo control que probaron "
        "al menos un producto nuevo."
    )
