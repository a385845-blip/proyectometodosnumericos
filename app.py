import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Calculadora de Métodos Numéricos", layout="wide")
st.title("🧮 Solucionador de Ecuaciones")
st.markdown("Encuentra raíces de funciones y analiza el error por iteración.")

# --- LÓGICA DE LOS MÉTODOS ---

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0: return None, "Error: f(a) y f(b) deben tener signos opuestos.", []
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
    if f(a) * f(b) >= 0: return None, "Error: f(a) y f(b) deben tener signos opuestos.", []
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
        if abs(dfxn) < 1e-12: return None, "Error: Derivada cercana a cero.", historial_error
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
        if abs(fx1 - fx0) < 1e-12: return None, "Error: División por cero.", historial_error
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x_next - x1)
        historial_error.append(error)
        if error < tol: return x_next, i + 1, historial_error
        x0, x1 = x1, x_next
    return x1, max_iter, historial_error

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
func_input = st.sidebar.text_input("Función f(x):", "np.exp(-x) - x")
metodo = st.sidebar.selectbox("Método:", ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante"])
tol = st.sidebar.number_input("Tolerancia:", value=1e-5, format="%.1e")
max_iter = st.sidebar.number_input("Iteraciones máximas:", value=50)

# Validación de la función
f_valid = False
try:
    f = lambda x: eval(func_input, {"np": np, "x": x})
    f(0) # Prueba rápida
    f_valid = True
except:
    st.sidebar.error("⚠️ Error en la función. Revisa la sintaxis.")

st.sidebar.markdown("---")
# Parámetros por método
if metodo in ["Bisección", "Falsa Posición"]:
    p_a = st.sidebar.number_input("Límite inferior (a):", value=0.0)
    p_b = st.sidebar.number_input("Límite superior (b):", value=2.0)
elif metodo == "Newton-Raphson":
    p_x0 = st.sidebar.number_input("Punto inicial (x0):", value=0.0)
    df_input = st.sidebar.text_input("Derivada f'(x):", "-np.exp(-x) - 1")
    df = lambda x: eval(df_input, {"np": np, "x": x})
else: # Secante
    p_x0 = st.sidebar.number_input("Punto x0:", value=0.0)
    p_x1 = st.sidebar.number_input("Punto x1:", value=1.0)

# --- EJECUCIÓN ---
if st.sidebar.button("🚀 Calcular"):
    if not f_valid:
        st.error("Por favor, corrige la función antes de continuar.")
    else:
        try:
            if metodo == "Bisección": res, iters, errores = biseccion(f, p_a, p_b, tol, max_iter)
            elif metodo == "Falsa Posición": res, iters, errores = falsa_posicion(f, p_a, p_b, tol, max_iter)
            elif metodo == "Newton-Raphson": res, iters, errores = newton_raphson(f, df, p_x0, tol, max_iter)
            else: res, iters, errores = secante(f, p_x0, p_x1, tol, max_iter)

            if res is None:
                st.error(iters)
            else:
                col1, col2 = st.columns([1, 1.2])
                
                with col1:
                    st.subheader("📊 Datos del Resultado")
                    st.success(f"**Raíz hallada:** {res:.8f}")
                    st.info(f"**Total iteraciones:** {iters}")
                    st.warning(f"**Error final estimado:** {errores[-1]:.2e}")
                    
                    # Tabla de errores generada con Pandas
                    df_err = pd.DataFrame({
                        "Iteración": range(1, len(errores)+1), 
                        "Error Estimado": errores
                    })
                    st.dataframe(df_err, use_container_width=True)

                with col2:
                    st.subheader("📈 Gráfica de la Función")
                    fig, ax = plt.subplots()
                    
                    # Definir rango de visualización alrededor de la raíz
                    x_vals = np.linspace(res-3, res+3, 300)
                    y_vals = [f(i) for i in x_vals]
                    
                    ax.plot(x_vals, y_vals, label=f'f(x)', color='#1f77b4', lw=2)
                    ax.axhline(0, color='black', lw=1) # Eje X
                    ax.axvline(0, color='black', lw=1) # Eje Y
                    ax.scatter([res], [0], color='red', s=80, label=f'Raíz: {res:.4f}', zorder=3)
                    
                    ax.set_title(f"Método de {metodo}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

# Botón de reinicio
st.markdown("---")
if st.button("🔄 Nueva Ecuación / Otro Método"):
    st.rerun()
