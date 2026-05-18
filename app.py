import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Calculadora de Métodos Numéricos")
st.title("Solucionador de Ecuaciones")
st.write("Encuentra raíces de funciones y analiza el error por iteración.")


# LÓGICA DE LOS MÉTODOS (FUNCIONES)

def biseccion(f, a, b, tol, max_iter):
    # Validar que cambie de signo
    if f(a) * f(b) >= 0:
        return None, "Error: f(a) y f(b) deben tener signos opuestos."
   
    historial_error = []
    c_ant = a
   
    for i in range(max_iter):
        c = (a + b) / 2
       
        # Calcular error
        if i == 0:
            error = abs(b - a)
        else:
            error = abs(c - c_ant)
           
        historial_error.append(error)
       
        # Condición de paro
        if error < tol or abs(f(c)) < 1e-12:
            return c, historial_error
           
        # Cambiar límites
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        c_ant = c
       
    return c, historial_error


def falsa_posicion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Error: f(a) y f(b) deben tener signos opuestos."
       
    historial_error = []
    c_ant = a
   
    for i in range(max_iter):
        # Fórmula de falsa posición
        c = b - (f(b) * (a - b)) / (f(a) - f(b))
       
        if i == 0:
            error = abs(b - a)
        else:
            error = abs(c - c_ant)
           
        historial_error.append(error)
       
        if error < tol:
            return c, historial_error
           
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        c_ant = c
       
    return c, historial_error


def newton_raphson(f, df, x0, tol, max_iter):
    xn = x0
    historial_error = []
   
    for i in range(max_iter):
        dfxn = df(xn)
        if abs(dfxn) < 1e-12:
            return None, "Error: Derivada cercana a cero."
           
        xn_siguiente = xn - f(xn) / dfxn
        error = abs(xn_siguiente - xn)
        historial_error.append(error)
       
        if error < tol:
            return xn_siguiente, historial_error
           
        xn = xn_siguiente
       
    return xn, historial_error


def secante(f, x0, x1, tol, max_iter):
    historial_error = []
   
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
       
        if abs(fx1 - fx0) < 1e-12:
            return None, "Error: División por cero."
           
        xn_siguiente = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(xn_siguiente - x1)
        historial_error.append(error)
       
        if error < tol:
            return xn_siguiente, historial_error
           
        x0 = x1
        x1 = xn_siguiente
       
    return x1, historial_error


# ==========================================
# INTERFAZ DE USUARIO (SIDEBAR)
# ==========================================

st.sidebar.header("Configuración")
func_input = st.sidebar.text_input("Función f(x):", "np.exp(-x) - x")
metodo = st.sidebar.selectbox("Método:", ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante"])
tol = st.sidebar.number_input("Tolerancia:", value=0.00001)
max_iter = st.sidebar.number_input("Iteraciones máximas:", value=50)

# Crear la función matemática usando una función normal en vez de lambdas complejas
def f(x):
    return eval(func_input)

st.sidebar.markdown("---")

# Pedir datos según el método seleccionado
if metodo == "Bisección" or metodo == "Falsa Posición":
    p_a = st.sidebar.number_input("Límite inferior (a):", value=0.0)
    p_b = st.sidebar.number_input("Límite superior (b):", value=2.0)
   
elif metodo == "Newton-Raphson":
    p_x0 = st.sidebar.number_input("Punto inicial (x0):", value=0.0)
    df_input = st.sidebar.text_input("Derivada f'(x):", "-np.exp(-x) - 1")
    def df(x):
        return eval(df_input)
       
else: # Secante
    p_x0 = st.sidebar.number_input("Punto x0:", value=0.0)
    p_x1 = st.sidebar.number_input("Punto x1:", value=1.0)


# ==========================================
# BOTÓN DE CÁLCULO Y RESULTADOS
# ==========================================

if st.sidebar.button("Calcular"):
    try:
        # Ejecutar el método correspondiente
        error_msg = None
       
        if metodo == "Bisección":
            resultado = biseccion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Falsa Posición":
            resultado = falsa_posicion(f, p_a, p_b, tol, max_iter)
        elif metodo == "Newton-Raphson":
            resultado = newton_raphson(f, df, p_x0, tol, max_iter)
        else:
            resultado = secante(f, p_x0, p_x1, tol, max_iter)
           
        # Desempaquetar el resultado (Si es una tupla con error o con datos válidos)
        if resultado[0] is None:
            st.error(resultado[1])
        else:
            raiz = resultado[0]
            errores = resultado[1]
            total_iter = len(errores)
           
            # Mostrar los textos de resultados
            st.subheader("Resultados:")
            st.write(f"**Raíz hallada:** {raiz}")
            st.write(f"**Total iteraciones:** {total_iter}")
            st.write(f"**Error final:** {errores[-1]}")
           
            # Crear y mostrar tabla
            st.subheader("Historial de Errores:")
            df_err = pd.DataFrame({
                "Iteración": range(1, total_iter + 1),
                "Error": errores
            })
            st.dataframe(df_err)
           
            # Hacer la gráfica
            st.subheader("Gráfica:")
            fig, ax = plt.subplots()
           
            # Rango de la gráfica basado en la raíz
            x_valores = np.linspace(raiz - 3, raiz + 3, 200)
            y_valores = []
            for xi in x_valores:
                y_valores.append(f(xi))
               
            ax.plot(x_valores, y_valores, label="f(x)", color="blue")
            ax.axhline(0, color="black") # Eje horizontal
            ax.axvline(0, color="black") # Eje vertical
            ax.plot(raiz, 0, marker="o", color="red", label=f"Raíz: {raiz:.4f}")
           
            ax.set_title("Gráfica de la función")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
           
    except Exception as e:
        st.error(f"Hubo un error en los datos o la función: {e}")

# Botón de reiniciar abajo
st.markdown("---")
if st.button("Limpiar todo"):
    st.rerun()
