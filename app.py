import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Calculadora Numérica Pro", layout="wide")
st.title("🧮 Calculadora de Métodos Numéricos")

# --- MÉTODOS CON CÁLCULO DE ERROR ---

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0: return None, "Cambio de signo no detectado.", []
    historial_error = []
    c_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(c - c_ant) if i > 0 else abs(b - a)
        historial_error.append(error)
        if error < tol or abs(f(c)) < 1e-12: return c, i + 1, historial_error
        if f(c) * f(a) < 0: b = c
        else: a = c
        c_ant = c
    return c, max_iter, historial_error

def falsa_posicion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0: return None, "Cambio de signo no detectado.", []
    historial_error = []
    c_ant = a
    for i in range(max_iter):
        c = b - (f(b) * (a - b)) / (f(a) - f(b))
        error = abs(c - c_ant) if i > 0 else abs(b - a)
        historial_error.append(error)
        if error < tol: return c, i + 1, historial_error
        if f(c) * f(a) < 0: b = c
        else: a = c
        c_ant = c
    return c, max_iter, historial_error

def newton_raphson(f, df, x0, tol, max_iter):
    xn = x0
    historial_error = []
    for i in range(max_iter):
        dfxn = df(xn)
        if abs(dfxn) < 1e-12: return None, "Derivada cero.", historial_error
        xn_next = xn - f(xn) / dfxn
        error = abs(xn_next - xn)
        historial_error.append(error)
        if error < tol: return xn_next, i + 1, historial_error
        xn = xn_next
    return xn, max_iter, historial_error

def secante(f, x0, x1, tol, max_iter):
    historial_error = []
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < 1e-12: return None, "División por cero.", historial_error
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x_next - x1)
        historial_error.append(error)
        if error < tol: return x_next, i + 1, historial_error
        x0, x1 = x1, x_next
    return x1, max_iter, historial_error

# --- INTERFAZ LATERAL ---
st.sidebar.header("Datos del problema")
func_input = st.sidebar.text_input("Función f(x):", "x**2 - 4 (ejemplo)")
metodo = st.sidebar.selectbox("Método:", ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante"])
tol = st.sidebar.number_input("Tolerancia:", value=1e-5, format="%.1e")
max_iter = st.sidebar.number_input("Iteraciones máx:", value=50)

# Parámetros específicos
st.sidebar.markdown("---")
f = lambda x: eval(func_input, {"np": np, "x": x})

if metodo in ["Bisección", "Falsa Posición"]:
    p_a = st.sidebar.number_input("a:", value=0.0)
    p_b = st.sidebar.number_input("b:", value=3.0)
elif metodo == "Newton-Raphson":
    p_x0 = st.sidebar.number_input("x0:", value=1.0)
    df_input = st.sidebar.text_input("f'(x):", "2*x")
    df = lambda x: eval(df_input, {"np": np, "x": x})
else:
    p_x0 = st.sidebar.number_input("x0:", value=1.0)
    p_x1 = st.sidebar.number_input("x1:", value=3.0)

# --- EJECUCIÓN ---
if st.sidebar.button("🚀 Calcular"):
    res, iters, errores = None, 0, []
    try:
        if metodo == "Bisección": res, iters, errores = biseccion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Falsa Posición": res, iters, errores = falsa_posicion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Newton-Raphson": res, iters, errores = newton_raphson(f, df, p_x0, tol, max_iter)
        else: res, iters, errores = secante(f, p_x0, p_x1, tol, max_iter)

        if res is None:
            st.error(f"Error: {iters}")
        else:
            # Layout de resultados
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Resultados")
                st.success(f"**Raíz:** {res:.8f}")
                st.info(f"**Iteraciones:** {iters}")
                st.warning(f"**Último Error:** {errores[-1]:.2e}")
                
                # Tabla de errores
                df_err = pd.DataFrame({"Iteración": range(1, len(errores)+1), "Error Estimado": errores})
                st.dataframe(df_err, use_container_width=True)

            with col2:
                st.subheader("📈 Gráficas")
                tab1, tab2 = st.tabs(["Función", "Convergencia (Error)"])
                
                with tab1:
                    fig1, ax1 = plt.subplots()
                    x_vals = np.linspace(res-2, res+2, 200)
                    ax1.plot(x_vals, [f(i) for i in x_vals], label='f(x)')
                    ax1.axhline(0, color='black', lw=1)
                    ax1.scatter([res], [0], color='red', label=f'Raíz: {res:.4f}')
                    ax1.legend(); ax1.grid(True)
                    st.pyplot(fig1)
                
                with tab2:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(range(1, len(errores)+1), errores, marker='o', color='green')
                    ax2.set_yscale('log') # Escala logarítmica para ver mejor el error
                    ax2.set_title("Caída del Error por Iteración")
                    ax2.set_xlabel("Iteración"); ax2.set_ylabel("Error (log)")
                    ax2.grid(True, which="both", ls="-")
                    st.pyplot(fig2)
    except Exception as e:
        st.error(f"Error matemático: {e}")

if st.button("🔄 Reiniciar"): st.rerun()
