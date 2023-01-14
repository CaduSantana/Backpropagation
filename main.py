from rede import Rede
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QLineEdit, QPushButton, QTextBrowser, QLabel, QFileDialog, QApplication
from PyQt5 import uic
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f_logistica(x):
    return 1 / (1 + np.exp(-x))
def f_logistica_derivada (x):
    return f_logistica(x) * (1 - f_logistica(x))
def f_hiperbolica (x):
    return (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
def f_hiperbolica_derivada (x):
    return 1 - f_hiperbolica(x)**2


class Interface(QMainWindow):
    def __init__(self) -> None:
        super(Interface, self).__init__()
        uic.loadUi('.\interface.ui', self)
        self.show()

        self.btn_logistica = self.findChild(QRadioButton, 'btnLogistica')
        self.btn_hiperbolica = self.findChild(QRadioButton, 'btnHiperbolica')
        self.f = None
        self.f_derivada = None
        
        self.btn_erro = self.findChild(QRadioButton, 'btnErro')
        self.btn_num_iteracoes = self.findChild(QRadioButton, 'btnNumIteracoes')
        self.input_erro = self.findChild(QLineEdit, 'inputErro')
        self.input_iteracoes = self.findChild(QLineEdit, 'inputIteracoes')

        self.btn_abrir_teste = self.findChild(QPushButton, 'btnAbrirTeste')
        self.btn_abrir_teste.clicked.connect(self.abrir_teste)
        self.btn_abrir_treino = self.findChild(QPushButton, 'btnAbrirTreino')
        self.btn_abrir_treino.clicked.connect(self.abrir_treino)
        self.btn_predizer = self.findChild(QPushButton, 'btnPredizer')
        self.btn_predizer.clicked.connect(self.predizer)
        self.btn_treinar = self.findChild(QPushButton, 'btnTreinar')
        self.btn_treinar.clicked.connect(self.treinar)

        self.input_oculta = self.findChild(QLineEdit, "inputNumOculta")
        self.btn_automatico = self.findChild(QRadioButton, "btnDefAuto")
        self.btn_inserir = self.findChild(QRadioButton, "btnInserir")

        self.arquivo_teste = None
        self.arquivo_treino = None

        self.saida_texto = self.findChild(QTextBrowser, 'saidaTexto')
        self.label_img = self.findChild(QLabel, 'labelImg')

        self.rede = None
        self.saida_rede = None
        self.saida_esperada = None
        self.matriz_confusao = None

    
    def abrir_teste(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        filename = dialog.getOpenFileName()
        self.arquivo_teste = filename[0]
        self.saida_texto.append("Arquivo de Teste aberto: " + self.arquivo_teste + "\n\n")

    
    def abrir_treino(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        filename = dialog.getOpenFileName()
        self.arquivo_treino = filename[0]
        self.saida_texto.append("Arquivo de Treino aberto: " + self.arquivo_treino + "\n\n")


    def treinar(self):
        self.saida_texto.append("----------------------------------------------------------------------------------\n" + 
                                "Carreagando Dados...\n")
        try:
            # abrir o arquivo
            df_entrada_treino = pd.read_csv(self.arquivo_treino)
        except:
            self.saida_texto.append("Erro ao abrir o arquivo de treino.\n")
            return
        # pegando o número de entradas
        num_entradas = df_entrada_treino.shape[1] - 1
        # pegando o número de saídas
        num_saidas = df_entrada_treino['classe'].max()
        num_oculta = self.calcula_oculta(num_entradas=num_entradas, num_saidas=num_saidas)

        self.saida_texto.append("Número de entradas (parâmetros): " + str(num_entradas) + 
                               "\nNúmero de ocultas: " + str(num_oculta) + 
                               "\nNúmero de saídas (classes): " + str(num_saidas) + 
                               "\n\n")

        self.saida_esperada = self.montar_vetor_saida_esperada(df_entrada_treino)
        entrada = df_entrada_treino.to_numpy()
        entrada = np.delete(entrada, df_entrada_treino.shape[1]-1, 1)

        self.definir_f()

        self.saida_texto.append("Treinamento iniciado.\n")
        self.rede = Rede(entrada=entrada, n_ocultos=num_oculta, n_saidas=num_saidas, saida_esperada=self.saida_esperada, f=self.f, f_derivada=self.f_derivada, taxa_aprendizado=1)
        
        if self.is_erro_max():
            erro_max = float(self.input_erro.text())
            self.rede.treinar(erro_max=erro_max, is_erro_max=True)
        else:
            iteracoes = int(self.input_iteracoes.text())
            self.rede.treinar(n_iteracoes=iteracoes)
        
        self.saida_texto.append("Erro da rede: " + str(self.rede.erro_rede) + "\n")
        self.saida_texto.append("Treinamento finalizado.\n\n")
        
        
    def predizer(self):
        self.saida_texto.append("----------------------------------------------------------------------------------\n" +
                                "Carreagando Dados...\n")
        try:
            # abrir o arquivo
            df_entrada_teste = pd.read_csv(self.arquivo_teste)
        except:
            self.saida_texto.append("Erro ao abrir o arquivo de teste.\n")
            return

        saida_esperada = self.montar_vetor_saida_esperada(df_entrada_teste)
        entrada = df_entrada_teste.to_numpy()
        entrada = np.delete(entrada, df_entrada_teste.shape[1] - 1, 1)

        self.saida_texto.append("Predição iniciada.\n")
        self.rede.prever(entrada)
        self.saida_rede = self.rede.tranf_saida
        self.saida_texto.append("Matriz de confusão:\n")
        self.monta_matriz_confusao(saida_esperada=saida_esperada)
        self.saida_texto.append(str(self.matriz_confusao) + "\n")
        print(self.saida_rede)
        self.saida_texto.append("Predição finalizada.\n\n")

    
    def montar_vetor_saida_esperada(self, df_entrada_treino):
        # saida_esperada = np.array([df_entrada_treino['classe']]).T
        saida_esperada = np.zeros((df_entrada_treino.shape[0], df_entrada_treino['classe'].max()))
        for i in range(df_entrada_treino.shape[0]):
            saida_esperada[i][df_entrada_treino['classe'][i] - 1] = 1

        return saida_esperada

    
    def monta_matriz_confusao(self, saida_esperada):
        self.matriz_confusao = np.zeros((self.saida_rede.shape[1], self.saida_rede.shape[1]))
        for rede, esperada in zip(self.saida_rede, saida_esperada):
            indice_esperado = np.argmax(esperada)
            indice_saida = np.argmax(rede)
            self.matriz_confusao[indice_saida][indice_esperado] += 1


    def calcula_oculta(self, num_entradas, num_saidas):
        if (self.btn_inserir.isChecked()):
            return int(self.input_oculta.text())
        elif (self.btn_automatico.isChecked()):
            return int(np.round(np.sqrt(num_entradas * num_saidas)))


    def is_erro_max(self):
        return self.btn_erro.isChecked()
    

    def definir_f(self):
        if (self.btn_logistica.isChecked()):
            self.f = f_logistica
            self.f_derivada = f_logistica_derivada
        else:
            self.f = f_hiperbolica
            self.f_derivada = f_hiperbolica_derivada


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Interface()
    app.exec_()
