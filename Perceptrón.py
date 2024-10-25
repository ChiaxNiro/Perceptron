import numpy as np
import matplotlib.pyplot as plt

# Función de activación escalón
def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

# Función softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Inicialización de pesos
def inicializar_pesos(entrada_dim, oculta_dim, salida_dim):
    np.random.seed(42)
    pesos_entrada_oculta = np.random.randn(entrada_dim, oculta_dim)
    pesos_oculta_salida = np.random.randn(oculta_dim, salida_dim)
    return pesos_entrada_oculta, pesos_oculta_salida

# Propagación hacia adelante (forward propagation)
def forward_propagation(x, pesos_entrada_oculta, pesos_oculta_salida):
    # Capa oculta
    z_oculta = np.dot(x, pesos_entrada_oculta)
    a_oculta = funcion_escalon(z_oculta)

    # Capa de salida
    z_salida = np.dot(a_oculta, pesos_oculta_salida)
    a_salida = softmax(z_salida)

    return a_salida

# Generar ejemplos de líneas (10x10 pixels)
def generar_lineas(n):
    ejemplos = []
    for _ in range(n):
        imagen = np.zeros((10, 10))
        linea = np.random.randint(0, 10)  # Línea aleatoria horizontal o vertical
        if np.random.rand() > 0.5:  # Horizontal
            imagen[linea, :] = 1
        else:  # Vertical
            imagen[:, linea] = 1
        ejemplos.append(imagen.flatten())  # Aplanamos la imagen 10x10 a un vector de 100
    return np.array(ejemplos)

# Generar ejemplos de círculos (10x10 pixels)
def generar_circulos(n):
    ejemplos = []
    for _ in range(n):
        imagen = np.zeros((10, 10))
        centro = (5, 5)
        radio = 4  # Aumentar un poco el radio para diferenciar mejor
        for i in range(10):
            for j in range(10):
                if (i - centro[0]) ** 2 + (j - centro[1]) ** 2 <= radio**2:  # Radio del círculo
                    imagen[i, j] = 1
        ejemplos.append(imagen.flatten())  # Aplanamos la imagen 10x10 a un vector de 100
    return np.array(ejemplos)

# Función de clasificación utilizando el perceptrón
def perceptron(ejemplos, pesos_entrada_oculta, pesos_oculta_salida):
    salidas = forward_propagation(ejemplos, pesos_entrada_oculta, pesos_oculta_salida)
    return np.argmax(salidas, axis=1) + 1  # Retorna 1 para líneas, 2 para círculos

# Calcular la precisión
def calcular_precision(predicciones, etiquetas):
    return np.mean(predicciones == etiquetas) * 100

# Mostrar imágenes generadas
def mostrar_imagenes(ejemplos, etiquetas, predicciones, num_imagenes=5):
    for i in range(num_imagenes):
        plt.subplot(2, num_imagenes, i+1)
        plt.imshow(ejemplos[i].reshape(10, 10), cmap='gray')
        plt.title(f"Etiqueta: {etiquetas[i]} \nPred: {predicciones[i]}")
        plt.axis('off')

    for i in range(num_imagenes):
        plt.subplot(2, num_imagenes, i+num_imagenes+1)
        plt.imshow(ejemplos[i+30].reshape(10, 10), cmap='gray')
        plt.title(f"Etiqueta: {etiquetas[i+30]} \nPred: {predicciones[i+30]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Función principal
def main():
    # Ajustamos la capa oculta a 50 neuronas para capturar mejor las características
    pesos_entrada_oculta, pesos_oculta_salida = inicializar_pesos(100, 50, 2)

    # Generar ejemplos de prueba
    ejemplos_lineas = generar_lineas(30)
    ejemplos_circulos = generar_circulos(30)

    # Concatenamos ejemplos y creamos las etiquetas correspondientes
    ejemplos = np.vstack((ejemplos_lineas, ejemplos_circulos))
    etiquetas = np.array([1] * 30 + [2] * 30)  # 1 para líneas, 2 para círculos

    # Clasificar ejemplos con el perceptrón
    predicciones = perceptron(ejemplos, pesos_entrada_oculta, pesos_oculta_salida)

    # Calcular precisión
    precision = calcular_precision(predicciones, etiquetas)

    # Imprimir resultados
    print(f"Precisión del modelo: {precision:.2f}%")
    print(f"Predicciones: {predicciones}")
    print(f"Etiquetas reales: {etiquetas}")

    # Mostrar imágenes y sus predicciones
    mostrar_imagenes(ejemplos, etiquetas, predicciones)

if __name__ == "__main__":
    main()