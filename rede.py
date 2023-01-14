import numpy as np


# Funções de tranferência
def f_logistica(x):
    return 1 / (1 + np.exp(-x))
def f_logistica_derivada (x):
    return f_logistica(x) * (1 - f_logistica(x))
def f_hiperbolica (x):
    return (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
def f_hiperbolica_derivada (x):
    return 1 - f_hiperbolica(x)**2


class Rede:
    def __init__(self, entrada: np.array, n_ocultos: int, n_saidas: int, saida_esperada: np.array, f, f_derivada, taxa_aprendizado = 1):
        # Definindo as variáveis da rede
        self.entrada = entrada
        self.saida = None
        self.n_ocultos = n_ocultos
        self.taxa_aprendizado = taxa_aprendizado
        self.saida_esperada = saida_esperada
        self.f = f
        self.f_derivada = f_derivada

        # Definindo pesos aleatórios para cada camada
        self.peso_oculta = np.random.random((entrada.shape[1], n_ocultos)) /100
        self.peso_saida = np.random.random((n_ocultos, n_saidas)) /100

        # Delta das Camadas
        self.delta_oculta = None
        self.delta_saida = None

        # Derivada Parcial das Camadas
        self.d_parcial_oculta = None
        self.d_parcial_saida = None

        # Valor de tranferencia das camadas
        self.tranf_oculta = None
        self.tranf_saida = None

        self.erro_rede = None
        self.matriz_confusao = np.zeros((n_saidas, n_saidas))
    

    def passagem_frente(self):
        # Passagem para frente --> Forward Propagation
        # 1. Camada Oculta
        oculta = np.dot(self.entrada, self.peso_oculta)
        self.tranf_oculta = self.f(oculta)
        self.d_parcial_oculta = self.f_derivada(self.tranf_oculta)
        # 2. Camada saída
        self.saida = np.dot(self.tranf_oculta, self.peso_saida)
        self.tranf_saida = self.f(self.saida)
        self.d_parcial_saida = self.f_derivada(self.tranf_saida)
        

    def passagem_tras(self):
        # Passagem para trás --> Back Propagarion
        # 1. Calcular erros
        # 1.1. Camada de saída
        erro_saida = (self.tranf_saida - self.saida_esperada)
        self.erro_rede = np.sum(erro_saida**2) /2
        self.delta_saida = np.atleast_2d(erro_saida * self.d_parcial_saida)
        # 1.2. Camada oculta
        self.delta_oculta = np.atleast_2d(self.d_parcial_oculta * np.dot(self.delta_saida, self.peso_saida.T))

        # 2. Atualizar pesos
        self.peso_saida -= np.dot(self.tranf_oculta.T, self.delta_saida) * self.taxa_aprendizado
        self.peso_oculta -= np.dot(self.entrada.T, self.delta_oculta) * self.taxa_aprendizado


    def treinar(self, n_iteracoes, erro_max = None, is_erro_max = False):
        if is_erro_max:
            while self.erro_rede > erro_max:
                self.passagem_frente()
                self.passagem_tras()
        else:
            for i in range(n_iteracoes):
                self.passagem_frente()
                self.passagem_tras()
    

    def prever(self, entrada):
        for _ in entrada:
            self.entrada = entrada
            self.passagem_frente()

        saida = self.tranf_saida.round()
        for i in saida:
            indice = np.where(i == 1)
            self.matriz_confusao[indice] += 1


if __name__ == '__main__':
    entrada = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1],
                        [0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1],
                        [0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    saida_esperada = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]]).T


    n_ocultos = 4
    n_saidas = 1
    n_iretacoes = 1000
    rede = Rede(entrada, n_ocultos, n_saidas, saida_esperada, f_logistica, f_logistica_derivada)

    rede.treinar(n_iretacoes)
    print("\n\n Predição: \n")
    rede.prever(entrada)
    print(rede.saida)
    