"""
lstm_model_numpy.py — LSTM implementada desde cero con NumPy
Backpropagation Through Time (BPTT) completo - VERSIÓN CORREGIDA
"""

import numpy as np
from typing import Tuple, List


class LSTMCell:
    """Celda LSTM individual con BPTT"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = np.sqrt(1.0 / hidden_size)
        
        # Pesos para las puertas
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_c = np.random.randn(hidden_size, hidden_size + input_size) * scale
        
        # Bias (forget gate bias en 1 para mejor aprendizaje)
        self.b_f = np.ones((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        
        # Cache para BPTT
        self.cache = None
        
        # Acumuladores de gradientes
        self.grads = None
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Forward pass: un paso temporal
        x_t: (input_size, batch_size)
        h_prev: (hidden_size, batch_size)
        c_prev: (hidden_size, batch_size)
        
        Returns:
            h_next: (hidden_size, batch_size)
            c_next: (hidden_size, batch_size)
            cache: dict con valores intermedios
        """
        # Concatenar entrada y hidden state anterior
        concat = np.vstack([h_prev, x_t])  # (hidden_size + input_size, batch_size)
        
        # Puertas
        f = self._sigmoid(self.W_f @ concat + self.b_f)  # forget gate
        i = self._sigmoid(self.W_i @ concat + self.b_i)  # input gate
        c_tilde = self._tanh(self.W_c @ concat + self.b_c)  # candidate
        o = self._sigmoid(self.W_o @ concat + self.b_o)  # output gate
        
        # Cell state y hidden state
        c_next = f * c_prev + i * c_tilde
        h_next = o * self._tanh(c_next)
        
        # Guardar cache para backward
        cache = {
            'concat': concat,
            'f': f,
            'i': i,
            'c_tilde': c_tilde,
            'o': o,
            'c_next': c_next,
            'c_prev': c_prev,
            'tanh_c_next': self._tanh(c_next)
        }
        
        self.cache = cache
        
        return h_next, c_next, cache
    
    def backward(self, dh_next, dc_next, cache):
        """
        Backward pass: un paso temporal
        dh_next: (hidden_size, batch_size)
        dc_next: (hidden_size, batch_size)
        cache: dict con valores del forward
        """
        concat = cache['concat']
        f = cache['f']
        i = cache['i']
        c_tilde = cache['c_tilde']
        o = cache['o']
        c_prev = cache['c_prev']
        tanh_c_next = cache['tanh_c_next']
        
        batch_size = concat.shape[1]
        
        # Gradiente de la output gate
        do = dh_next * tanh_c_next * (o * (1 - o))
        
        # Gradiente del cell state (dos fuentes)
        dc = dc_next + dh_next * o * (1 - tanh_c_next ** 2)
        
        # Gradientes de las puertas
        df = dc * c_prev * (f * (1 - f))
        di = dc * c_tilde * (i * (1 - i))
        dc_tilde = dc * i * (1 - c_tilde ** 2)
        
        # Gradiente hacia el cell state anterior
        dc_prev = dc * f
        
        # Gradientes de los pesos
        dW_f = (df @ concat.T) / batch_size
        db_f = np.sum(df, axis=1, keepdims=True) / batch_size
        
        dW_i = (di @ concat.T) / batch_size
        db_i = np.sum(di, axis=1, keepdims=True) / batch_size
        
        dW_c = (dc_tilde @ concat.T) / batch_size
        db_c = np.sum(dc_tilde, axis=1, keepdims=True) / batch_size
        
        dW_o = (do @ concat.T) / batch_size
        db_o = np.sum(do, axis=1, keepdims=True) / batch_size
        
        # Gradiente hacia el concat
        dconcat = (self.W_f.T @ df + self.W_i.T @ di + 
                   self.W_c.T @ dc_tilde + self.W_o.T @ do)
        
        # Separar dh_prev y dx
        dh_prev = dconcat[:self.hidden_size, :]
        
        # Guardar gradientes para actualización
        self.grads = {
            'W_f': dW_f, 'b_f': db_f,
            'W_i': dW_i, 'b_i': db_i,
            'W_c': dW_c, 'b_c': db_c,
            'W_o': dW_o, 'b_o': db_o
        }
        
        return dh_prev, dc_prev
    
    def update(self, learning_rate, clip=5.0):
        """Actualizar pesos con gradient clipping"""
        if self.grads is None:
            return
        for key in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o']:
            if key in self.grads:
                grad = np.clip(self.grads[key], -clip, clip)
                setattr(self, key, getattr(self, key) - learning_rate * grad)


class LSTMModel:
    """Modelo LSTM completo con capa de salida"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.cell = LSTMCell(input_size, hidden_size)
        
        # Capa de salida
        scale = np.sqrt(1.0 / hidden_size)
        self.W_out = np.random.randn(output_size, hidden_size) * scale
        self.b_out = np.zeros((output_size, 1))
        
        self.cache = None

        # ✅ NUEVO: peso para el sigmoide de salida
        self.W_out_sigmoid = np.random.randn(output_size, hidden_size) * scale
        self.b_out_sigmoid = np.zeros((output_size, 1))
    
    def forward_sequence(self, X_seq):
        """
        Forward pass de secuencia completa
        X_seq: (seq_len, input_size, batch_size)
        
        Returns:
            y_pred: (output_size, batch_size)
            caches: lista de caches de cada paso
        """
        seq_len, _, batch_size = X_seq.shape
        
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))
        
        caches = []
        
        for t in range(seq_len):
            h, c, cache = self.cell.forward(X_seq[t], h, c)
            caches.append(cache)
        
        # ✅ Salida con SIGMOIDE (garantiza 0-1)
        z = self.W_out @ h + self.b_out
        y_pred = 1.0 / (1.0 + np.exp(-z))  # Sigmoide
        
        self.cache = {'caches': caches, 'h_T': h}
        
        return y_pred, caches
    
    def backward_sequence(self, dy_pred, caches):
        """
        Backward pass completo (BPTT)
        dy_pred: (output_size, batch_size)
        caches: lista de caches de cada paso
        """
        batch_size = dy_pred.shape[1]
        
        # Gradiente de la capa de salida
        h_T = self.cache['h_T']
        dW_out = (dy_pred @ h_T.T) / batch_size
        db_out = np.sum(dy_pred, axis=1, keepdims=True) / batch_size
        
        # Gradiente que entra a la LSTM
        dh_next = self.W_out.T @ dy_pred
        dc_next = np.zeros_like(dh_next)
        
        # Acumuladores de gradientes
        grads_acum = {
            'W_f': np.zeros_like(self.cell.W_f),
            'b_f': np.zeros_like(self.cell.b_f),
            'W_i': np.zeros_like(self.cell.W_i),
            'b_i': np.zeros_like(self.cell.b_i),
            'W_c': np.zeros_like(self.cell.W_c),
            'b_c': np.zeros_like(self.cell.b_c),
            'W_o': np.zeros_like(self.cell.W_o),
            'b_o': np.zeros_like(self.cell.b_o),
        }
        
        # Backward a través del tiempo
        for cache in reversed(caches):
            dh_next, dc_next = self.cell.backward(dh_next, dc_next, cache)
            # Acumular gradientes
            if self.cell.grads is not None:
                for key in grads_acum:
                    grads_acum[key] += self.cell.grads[key]
        
        return grads_acum, {'dW_out': dW_out, 'db_out': db_out}
    
    def predict(self, X):
        """Predecir para una secuencia"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        X_seq = np.transpose(X, (1, 2, 0))  # (seq_len, features, batch)
        y_pred, _ = self.forward_sequence(X_seq)
        return y_pred.T  # (batch, output)
    
    def train(self, X, y, epochs=100, learning_rate=0.001, 
              batch_size=16, clip=5.0, verbose=True):
        """
        Entrenar el modelo con BPTT
        X: (num_samples, seq_len, features)
        y: (num_samples, output_size)
        
        Returns:
            losses: Lista de pérdidas por época
        """
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        
        # Asegurar que y tenga 2 dimensiones
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                current_batch_size = end - start
                
                X_batch = X_shuffled[start:end]  # (batch, seq_len, features)
                y_batch = y_shuffled[start:end]  # (batch, output)
                
                # Reorganizar para LSTM: (seq_len, features, batch)
                X_seq = np.transpose(X_batch, (1, 2, 0))
                
                # Forward
                y_pred, caches = self.forward_sequence(X_seq)
                # y_pred: (output_size, batch)
                
                # Calcular pérdida
                y_true = y_batch.T  # (output_size, batch)
                loss = np.mean((y_pred - y_true) ** 2)
                epoch_loss += loss * current_batch_size
                
                # Gradiente de la pérdida
                dy_pred = 2 * (y_pred - y_true) / current_batch_size
                
                # Backward
                grads_acum, out_grads = self.backward_sequence(dy_pred, caches)
                
                # Actualizar capa de salida
                self.W_out -= learning_rate * out_grads['dW_out']
                self.b_out -= learning_rate * out_grads['db_out']
                
                # Actualizar LSTM
                for key in grads_acum:
                    grad = np.clip(grads_acum[key], -clip, clip)
                    setattr(self.cell, key, getattr(self.cell, key) - learning_rate * grad)
            
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1}/{epochs}, Pérdida: {avg_loss:.6f}")
        
        return losses