import numpy as np
import matplotlib.pyplot as plt

# PERCEPTRON
def perceptron_activation(summation):
    return 1 if summation >= 0 else 0

def perceptron(inputs, weights, bias):
    summation = np.dot(inputs, weights) + bias
    return perceptron_activation(summation)

# LEER CSV(EXCEL CON LOS DATOS)
def read_data(file):
    data = np.genfromtxt(file, delimiter=',')
    inputs = data[:, :-1]
    outputs = data[:, -1]
    return inputs, outputs

# ENTRENAMIENTO DEL PERCEPTRON
def train_perceptron(inputs, outputs, learning_rate, max_epochs, convergence_criterion):
    num_inputs = inputs.shape[1]
    num_patterns = inputs.shape[0]
    
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    epochs = 0
    convergence = False

    while epochs < max_epochs and not convergence:
        convergence = True
        for i in range(num_patterns):
            input_pattern = inputs[i]
            output_prediction = outputs[i]
            output_received = np.dot(weights, input_pattern) + bias
            error = output_prediction - output_received
            
            if abs(error) > convergence_criterion:
                convergence = False
                weights += learning_rate * error * input_pattern
                bias += learning_rate * error
        epochs += 1
    return weights, bias

# TEST
def test_perceptron(inputs, weights, bias):
    output_received = np.dot(inputs, weights) + bias
    return np.vectorize(perceptron_activation)(output_received)

# CALCULO DE PRECISIÓN
def calculate_accuracy(outputs_real, outputs_predictions):
    correct_predictions = np.sum(outputs_real == outputs_predictions)
    total_predictions = len(outputs_real)
    accuracy = correct_predictions / total_predictions
    return accuracy

def plot_graph(inputs, outputs, weights, bias):
    plt.figure(figsize=(8, 6))
    # GRAFICAR PATRONES
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, s=100)
    
    # GRAFICAR RECTA
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = test_perceptron(np.c_[xx.ravel(), yy.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0])
    plt.title('Patrones y Recta Separadora')
    plt.xlabel('Entrada X1')
    plt.ylabel('Entrada X2')
    plt.grid(True)
    plt.show()

# MAIN
if __name__ == "__main__":
    training_file = 'XOR_trn.csv'
    test_file = 'XOR_tst.csv'

    inputs_train, outputs_train = read_data(training_file)
    inputs_test, outputs_test = read_data(test_file)

    # PARAMETROS
    max_epochs = 100
    learning_rate = 0.1
    convergence_criterion = 0.01  # ALTERACIONES ALEATORIAS 5%
    
    # ENTRENAMIENTO
    trained_weights, trained_bias = train_perceptron(inputs_train, outputs_train, learning_rate, 
    max_epochs, convergence_criterion)
    print("Perceptrón entrenado con éxito.")
    # PERCEPTRON CON DATOS DE ENTRADA
    outputs_predictions = test_perceptron(inputs_test, trained_weights, trained_bias)
    
    # SACAR PRESICIÓN
    accuracy = calculate_accuracy(outputs_test, outputs_predictions)
    print("Precisión del perceptrón en datos de prueba (Accuracy):", accuracy)
    
    # RESULTADOS
    print("Salidas en prueba:")
    print(outputs_test)
    print("Salidas predichas por el perceptrón:")
    print(outputs_predictions)

    plot_graph(inputs_train, outputs_train, trained_weights, trained_bias)
