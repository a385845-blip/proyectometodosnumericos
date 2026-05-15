import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Solucionador Numérico", layout="wide")

st.title("🧮 Calculadora de Métodos Numéricos")
st.markdown("""
Esta aplicación resuelve ecuaciones no lineales del tipo $f(x) = 0$ utilizando diferentes métodos iterativos.
""")

# --- DEFINICIÓN DE MÉTODOS ---

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Error: f(a) y f(b) deben tener signos opuestos (Cambio de signo no detectado)."
    
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
        try:
            fxn = f(xn)
            dfxn = df(xn)
            if abs(dfxn) < 1e-12:
                return None, "Error: La derivada es cero. El método no puede continuar."
            xn_next = xn - fxn / dfxn
            if abs(xn_next - xn) < tol:
                return xn_next, i + 1
            xn = xn_next
        except Exception as e:
            return None, f"Error en el cálculo: {e}"
    return xn, max_iter

def secante(f, x0, x1, tol, max_iter):
    for i in range(max_iter):
        try:
            fx0, fx1 = f(x0), f(x1)
            if abs(fx1 - fx0) < 1e-12:
                return None, "Error: División por cero (f(x1) - f(x0) es igual)."
            x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if abs(x_next - x1) < tol:
                return x_next, i + 1
            x0, x1 = x1, x_next
        except Exception as e:
            return None, f"Error en el cálculo: {e}"
    return x1, max_iter

# --- BARRA LATERAL (ENTRADA DE DATOS) ---

st.sidebar.header("1. Configurar Función")
func_input = st.sidebar.text_input("Defina f(x):", value="x**2 - 4")

# Validación segura de la función
f_valid = False
try:
    # Usamos un diccionario para permitir funciones de numpy
    safe_dict = {"np": np, "x": 0}
    eval(func_input, {"__builtins__": None}, safe_dict)
    f = lambda x: eval(func_input, {"np": np, "x": x})
    f_valid = True
except:
    st.sidebar.error("⚠️ Error de sintaxis en la función.")

st.sidebar.header("2. Seleccionar Método")
metodo = st.sidebar.selectbox("Método:", 
    ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante"])

st.sidebar.header("3. Parámetros")
tol = st.sidebar.number_input("Tolerancia:", value=1e-5, format="%.1e")
max_iter = st.sidebar.number_input("Iteraciones máx:", value=50)

# Entradas según el método seleccionado
st.sidebar.markdown("---")
if metodo in ["Bisección", "Falsa Posición"]:
    p_a = st.sidebar.number_input("Límite inferior (a):", value=0.0)
    p_b = st.sidebar.number_input("Límite superior (b):", value=3.0)
elif metodo == "Newton-Raphson":
    p_x0 = st.sidebar.number_input("Punto inicial (x0):", value=1.0)
    df_input = st.sidebar.text_input("Derivada f'(x):", value="2*x")
    df = lambda x: eval(df_input, {"np": np, "x": x})
else: # Secante
    p_x0 = st.sidebar.number_input("Punto inicial (x0):", value=1.0)
    p_x1 = st.sidebar.number_input("Punto inicial (x1):", value=3.0)

# --- BOTÓN DE EJECUCIÓN Y RESULTADOS ---

if st.sidebar.button("🚀 Calcular"):
    if not f_valid:
        st.error("Por favor, corrige la función antes de calcular.")
    else:
        resultado = None
        error_msg = ""
        
        # Ejecución del método
        if metodo == "Bisección":
            resultado, iteraciones = biseccion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Falsa Posición":
            resultado, iteraciones = falsa_posicion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Newton-Raphson":
            resultado, iteraciones = newton_raphson(f, df, p_x0, tol, max_iter)
        elif metodo == "Secante":
            resultado, iteraciones = secante(f, p_x0, p_x1, tol, max_iter)

        # Mostrar resultados
        if resultado is None or isinstance(resultado, str):
            st.error(f"No se pudo hallar la raíz. {iteraciones}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Raíz encontrada:** {resultado:.6f}")
                st.info(f"**Iteraciones:** {iteraciones}")
                st.metric("f(raiz)", f"{f(resultado):.2e}")

            # --- GRÁFICA ---
            with col2:
                fig, ax = plt.subplots()
                # Crear rango dinámico para la gráfica
                x_range = np.linspace(resultado - 5, resultado + 5, 500)
                y_range = [f(i) for i in x_range]
                
                ax.plot(x_range, y_range, label='f(x)', color='blue')
                ax.axhline(0, color='black', linewidth=1)
                ax.axvline(0, color='black', linewidth=1)
                ax.scatter([resultado], [0], color='red', s=100, label=f'Raíz: {resultado:.4f}')
                ax.set_title(f"Visualización: {metodo}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

# --- OPCIONES DE REPETICIÓN ---
st.markdown("---")
if st.button("🔄 Limpiar y reiniciar"):
    st.rerun()
