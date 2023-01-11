import numpy as np

# Função logística
f = lambda x: 1 / (1 + np.exp(-x))

# Derivada da função logística
f_derivada = lambda x: f(x) * (1 - f(x))

# Função hiperbólica
f2 = lambda x: (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))

# Derivada da função hiperbólica
f2_derivada = lambda x: 1 - f2(x)**2

# Inicializa os pesos com valores aleatórios
num_classes = 3
num_entradas = 2
num_saidas = num_classes
num_oculta = 4
peso_oculta = np.random.rand(num_entradas, num_oculta) / 100
peso_saida = np.random.rand(num_oculta, num_saidas) / 100

# Aplica o vetor de entradas X, de 1 à N:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Calcula-se os nets dos neurônios da camada oculta, para cada j entre 1 e L:
net_oculta = np.dot(X, peso_oculta)
nets_o_j = np.sum(net_oculta, axis=1)

# Aplica a função de transferência para obter as saídas i da camada oculta:
saida_oculta = f(nets_o_j)

# Calcula-se os nets dos neurônios da camada de saída, para cada k entre 1 e M:
net_saida = np.dot(saida_oculta, peso_saida)
nets_s_k = np.sum(net_saida)

# Aplica a função de transferência para obter as saídas i da camada de saída:
omicron_k = f(nets_s_k)

# Calcula-se o erro da camada de saída:
erro_saida_s_k = (1 - omicron_k) * f_derivada(nets_s_k) # (1 - omicron_k) é o delta_k

# Calcula-se o erro da camada oculta:
erro_oculta_o_j = f_derivada(nets_o_j) * np.dot(erro_saida_s_k, peso_saida.T)

# # Atualiza os pesos da camada de saída:
# peso_saida = peso_saida + np.dot(saida_oculta.T, erro_saida_s_k)

# # Atualiza os pesos da camada oculta:
# peso_oculta = peso_oculta + np.dot(X.T, erro_oculta_o_j)

# Imprime os pesos atualizados
print("Pesos atualizados da camada oculta: ", peso_oculta)

print("Pesos atualizados da camada de saída: ", peso_saida)

# Imprime os valores de saída
print("Saída da rede: ", omicron_k)

# Calcula o erro da rede:
erro_rede = 1/2 * np.sum(erro_saida_s_k**2)

# Imprime o erro da rede
print("Erro da rede: ", erro_rede)