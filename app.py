import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración de la Página ---
st.set_page_config(page_title="Solucionador de Ecuaciones", layout="wide")
st.title("🧮 Calculadora de Métodos Numéricos")
st.write("Desarrollado para encontrar raíces de funciones de forma interactiva.")

# --- Funciones de los Métodos ---

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Error: f(a) y f(b) deben tener signos opuestos."
    
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, max_iter

def falsa_posicion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Error: f(a) y f(b) deben tener signos opuestos."
    
    for i in range(max_iter):
        c = b - (f(b) * (a - b)) / (f(a) - f(b))
        if abs(f(c)) < tol:
            return c, i + 1
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, max_iter

def newton_raphson(f, df, x0, tol, max_iter):
    xn = x0
    for i in range(max_iter):
        fxn = f(xn)
        dfxn = df(xn)
        if abs(dfxn) < 1e-10:
            return None, "Error: Derivada cercana a cero."
        xn_next = xn - fxn / dfxn
        if abs(xn_next - xn) < tol:
            return xn_next, i + 1
        xn = xn_next
    return xn, max_iter

def secante(f, x0, x1, tol, max_iter):
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < 1e-10:
            return None, "Error: División por cero (f(x1) - f(x0) muy pequeño)."
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x_next - x1) < tol:
            return x_next, i + 1
        x0, x1 = x1, x_next
    return x1, max_iter

# --- Interfaz Lateral (Entrada de Datos) ---

st.sidebar.header("Configuración de la Ecuación")
func_str = st.sidebar.text_input("Función f(x) (usar 'np' para funciones math)", "np.exp(-x) - x")
try:
    f = lambda x: eval(func_str)
    # Prueba rápida para validar la función
    f(1.0)
except:
    st.sidebar.error("Error en la sintaxis de la función.")

metodo = st.sidebar.selectbox("Selecciona el Método", 
    ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante"])

tol = st.sidebar.number_input("Tolerancia", value=1e-5, format="%.1e")
max_iter = st.sidebar.number_input("Iteraciones Máximas", value=100)

# --- Entradas específicas por método ---
st.sidebar.markdown("---")
if metodo in ["Bisección", "Falsa Posición"]:
    a = st.sidebar.number_input("Punto A (Izquierda)", value=0.0)
    b = st.sidebar.number_input("Punto B (Derecha)", value=1.0)
elif metodo == "Newton-Raphson":
    x0 = st.sidebar.number_input("Punto Inicial (x0)", value=0.0)
    df_str = st.sidebar.text_input("Derivada f'(x)", "-np.exp(-x) - 1")
    df = lambda x: eval(df_str)
else: # Secante
    x0 = st.sidebar.number_input("Punto x0", value=0.0)
    x1 = st.sidebar.number_input("Punto x1", value=1.0)

# --- Lógica de Cálculo y Visualización ---

if st.sidebar.button("Calcular Raíz"):
    res, info = None, ""
    
    with st.spinner('Calculando...'):
        if metodo == "Bisección":
            res, info = biseccion(f, a, b, tol, max_iter)
        elif metodo == "Falsa Posición":
            res, info = falsa_posicion(f, a, b, tol, max_iter)
        elif metodo == "Newton-Raphson":
            res, info = newton_raphson(f, df, x0, tol, max_iter)
        elif metodo == "Secante":
            res, info = secante(f, x0, x1, tol, max_iter)

    if isinstance(res, str) or res is None:
        st.error(f"Ocurrió un problema: {info}")
    else:
        st.success(f"**Resultado:** La raíz aproximada es **{res:.6f}** encontrada en **{info}** iteraciones.")
        
        # --- Gráfica ---
        fig, ax = plt.subplots()
        x_vals = np.linspace(res-2, res+2, 400)
        y_vals = [f(val) for val in x_vals]
        
        ax.plot(x_vals, y_vals, label=f"$f(x) = {func_str}$", color='blue')
        ax.axhline(0, color='black', lw=1)
        ax.scatter([res], [0], color='red', zorder=5, label=f"Raíz: {res:.4f}")
        ax.set_title(f"Visualización del Método: {metodo}")
        ax.legend()
        ax.grid(True, linestyle='--')
        
        st.pyplot(fig)

# --- Botón para Limpiar/Reiniciar ---
if st.button("Reiniciar Aplicación"):
    st.rerun()
