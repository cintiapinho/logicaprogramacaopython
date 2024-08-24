import numpy as np
import tkinter as tk
from tkinter import messagebox

# Rede Neural para o jogo da velha
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.bias_output = np.random.randn(self.output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backpropagation(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
        
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate
    
    def train(self, X, y, iterations=10000):
        for _ in range(iterations):
            output = self.forward(X)
            self.backpropagation(X, y, output)

# Função para converter o estado do tabuleiro em vetor
def board_to_input(board):
    return np.array(board).flatten()

# Função para treinar a rede neural
def train_nn(nn):
    # Exemplo simples de treinamento
    training_inputs = np.array([
        board_to_input([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # Jogador X em (0, 0)
        board_to_input([0, 1, 0, 0, 0, 0, 0, 0, 0]),  # Jogador X em (0, 1)
        board_to_input([0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Jogador X em (0, 2)
    ])
    
    training_outputs = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Melhor movimento seria (0, 1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Melhor movimento seria (0, 2)
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Melhor movimento seria (1, 1)
    ])
    
    nn.train(training_inputs, training_outputs)

# Função para a IA escolher um movimento
def ai_move(board, nn):
    input_board = board_to_input(board)
    output = nn.forward(input_board)
    move = np.argmax(output)
    return move

# Classe para a Interface Gráfica do Jogo da Velha
class TicTacToeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Jogo da Velha")
        self.board = [0] * 9  # Tabuleiro inicial vazio
        self.buttons = []
        self.current_player = 1  # 1 para jogador X, -1 para O
        self.nn = NeuralNetwork(9, 18, 9)  # Inicializa a rede neural
        
        train_nn(self.nn)  # Treina a rede neural antes do jogo começar
        
        self.create_board()
    
    def create_board(self):
        for i in range(9):
            button = tk.Button(self.master, text='', font='normal 20 bold', width=5, height=2,
                               command=lambda i=i: self.player_move(i))
            button.grid(row=i//3, column=i%3)
            self.buttons.append(button)
    
    def player_move(self, index):
        if self.board[index] == 0:  # Se a célula estiver vazia
            self.board[index] = self.current_player
            self.update_board()
            if self.check_winner(self.current_player):
                messagebox.showinfo("Fim de Jogo", "Voce venceu!")
                self.reset_game()
            elif 0 not in self.board:
                messagebox.showinfo("Fim de Jogo", "Empate!")
                self.reset_game()
            else:
                self.current_player = -1
                self.ai_turn()
    
    def ai_turn(self):
        move = ai_move(self.board, self.nn)
        if self.board[move] == 0:
            self.board[move] = self.current_player
            self.update_board()
            if self.check_winner(self.current_player):
                messagebox.showinfo("Fim de Jogo", "A IA venceu!")
                self.reset_game()
            elif 0 not in self.board:
                messagebox.showinfo("Fim de Jogo", "Empate!")
                self.reset_game()
            else:
                self.current_player = 1
    
    def update_board(self):
        for i in range(9):
            if self.board[i] == 1:
                self.buttons[i].config(text='X', state='disabled')
            elif self.board[i] == -1:
                self.buttons[i].config(text='O', state='disabled')
    
    #condição para as vitórias
    def check_winner(self, player):
        win_conditions = [(0, 1, 2), 
                          (3, 4, 5), 
                          (6, 7, 8),
                          (0, 3, 6), 
                          (1, 4, 7), 
                          (2, 5, 8),
                          (0, 4, 8), 
                          (2, 4, 6)]
        return any(all(self.board[pos] == player for pos in condition) for condition in win_conditions)
    
    def reset_game(self):
        self.board = [0] * 9
        for button in self.buttons:
            button.config(text='', state='normal')
        self.current_player = 1

# Executar o jogo
root = tk.Tk()
game = TicTacToeGame(root)
root.mainloop()
