import numpy as np

# Inicializa os pesos com valores aleatórios
def init_weights (M1 , M2 ):
    # M1 é o número de neurônios na camada de entrada
    # M2 é o número de neurônios na camada de saída
    # retorna uma matriz (M2 x (M1 + 1)) com valores aleatórios
    return np.random.randn (M2 , M1 + 1 ) / np.sqrt (M1 )

# Aplica o vetor de entradas x à rede neural
def forward (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M3) com as saídas da rede
    N , M1 = X.shape
    X = np.concatenate (( np.ones ((N , 1 )), X ), axis = 1 )
    Z = np.tanh (X.dot (W1 .T))
    Z = np.concatenate (( np.ones ((N , 1 )), Z ), axis = 1 )
    Y = Z.dot (W2 .T)
    return Y

# Calcula os nets dos neurônios da camada oculta, para cada j entre 1 e L:
def calc_nets (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M2) com os nets da camada oculta
    N , M1 = X.shape
    X = np.concatenate (( np.ones ((N , 1 )), X ), axis = 1 )
    Z = np.tanh (X.dot (W1 .T))
    return Z

# Aplica a função de transferência para obter as saídas i da camada oculta, para cada j:
def calc_outputs (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M2) com as saídas da camada oculta
    N , M1 = X.shape
    X = np.concatenate (( np.ones ((N , 1 )), X ), axis = 1 )
    Z = np.tanh (X.dot (W1 .T))
    Z = np.concatenate (( np.ones ((N , 1 )), Z ), axis = 1 )
    return Z

# Calcula os nets dos neurônios da camada oculta, para cada k entre 1 e M:
def calc_nets2 (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M3) com os nets da camada de saída
    N , M1 = X.shape
    X = np.concatenate (( np.ones ((N , 1 )), X ), axis = 1 )
    Z = np.tanh (X.dot (W1 .T))
    Z = np.concatenate (( np.ones ((N , 1 )), Z ), axis = 1 )
    Y = Z.dot (W2 .T)
    return Y

# Calcula as saídas O dos neurônios da camada de saída, para cada k:
def calc_outputs2 (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M3) com as saídas da camada de saída
    N , M1 = X.shape
    X = np.concatenate (( np.ones ((N , 1 )), X ), axis = 1 )
    Z = np.tanh (X.dot (W1 .T))
    Z = np.concatenate (( np.ones ((N , 1 )), Z ), axis = 1 )
    Y = Z.dot (W2 .T)
    return Y

# Calcula os erros para os neurônios da camada de saída:
def calc_errors2 (Y , T ):
    # Y é um vetor de saídas com dimensão (N x M3)
    # T é um vetor de saídas desejadas com dimensão (N x M3)
    # retorna uma matriz (N x M3) com os erros da camada de saída
    return Y - T

# Calcula os erros para os neurônios da camada oculta:
def calc_errors (Z , W2 , E2 ):
    # Z é um vetor de saídas com dimensão (N x M2)
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # E2 é um vetor de erros com dimensão (N x M3)
    # retorna uma matriz (N x M2) com os erros da camada oculta
    return (1 - Z ** 2 ) * E2 .dot (W2 [: , 1 :])

# Calcula o gradiente da função de custo em relação aos pesos da camada de saída:
def calc_grad2 (Z , E2 ):
    # Z é um vetor de saídas com dimensão (N x M2)
    # E2 é um vetor de erros com dimensão (N x M3)
    # retorna uma matriz (M2 x (M3 + 1)) com os gradientes da camada de saída
    N , M2 = Z.shape
    return E2 .T.dot (Z ) / N

# Calcula o gradiente da função de custo em relação aos pesos da camada oculta:
def calc_grad (X , E , Z ):
    # X é um vetor de entradas com dimensão (N x M1)
    # E é um vetor de erros com dimensão (N x M2)
    # Z é um vetor de saídas com dimensão (N x M2)
    # retorna uma matriz (M1 x (M2 + 1)) com os gradientes da camada oculta
    N , M1 = X.shape
    return E .T.dot (X ) / N

# Atualiza os pesos da camada de saída:
def update_weights2 (W2 , grad2 , eta ):
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # grad2 é um vetor de gradientes com dimensão (M2 x (M3 + 1))
    # eta é a taxa de aprendizado
    # retorna uma matriz (M2 x (M3 + 1)) com os novos pesos da camada de saída
    return W2 - eta * grad2

# Calcula o erro da rede:
def calc_error (Y , T ):
    # Y é um vetor de saídas com dimensão (N x M3)
    # T é um vetor de saídas desejadas com dimensão (N x M3)
    # retorna um escalar com o erro da rede
    return np.mean ((Y - T ) ** 2 )

# Abaixo segue o código para treinar a rede:
def train (X , T , M2 , eta , epochs ):
    # X é um vetor de entradas com dimensão (N x M1)
    # T é um vetor de saídas desejadas com dimensão (N x M3)
    # M2 é o número de neurônios da camada oculta
    # eta é a taxa de aprendizado
    # epochs é o número de épocas de treinamento
    N , M1 = X.shape
    N , M3 = T.shape
    W1 = np.random .rand (M1 + 1 , M2 ) * 2 - 1
    W2 = np.random .rand (M2 + 1 , M3 ) * 2 - 1
    for epoch in range (epochs ):
        Z = calc_outputs (X , W1 , W2 )
        Y = calc_outputs2 (X , W1 , W2 )
        E2 = calc_errors2 (Y , T )
        E = calc_errors (Z , W2 , E2 )
        grad2 = calc_grad2 (Z , E2 )
        grad = calc_grad (X , E , Z )
        W2 = update_weights2 (W2 , grad2 , eta )
        # W1 = update_weights (W1 , grad , eta )
        error = calc_error (Y , T )
        print ( 'Epoch: %d, MSE: %.3f' % (epoch , error ))
    return W1 , W2

# Abaixo segue o código para testar a rede:
def test (X , W1 , W2 ):
    # X é um vetor de entradas com dimensão (N x M1)
    # W1 é uma matriz de pesos com dimensão (M1 x (M2 + 1))
    # W2 é uma matriz de pesos com dimensão (M2 x (M3 + 1))
    # retorna uma matriz (N x M3) com as saídas da rede
    return calc_outputs2 (X , W1 , W2 )

X = np.array ([[ 0 , 0 ], [ 0 , 1 ], [ 1 , 0 ], [ 1 , 1 ]])
T = np.array ([[ 0 ], [ 1 ], [ 1 ], [ 0 ]])

train(X, T, 10, 0.01, 1000)