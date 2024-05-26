import torch.nn as nn
# pytorch mlp model, can set the number of layers and the number of neurons in each layer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(
                x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
