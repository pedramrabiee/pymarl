import torch.nn as nn


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=64, num_layers=2,
                 unit_activation=nn.ReLU, out_activation=nn.Identity,
                 batch_norm=False, layer_norm=False, batch_norm_first_layer=False):

        super(MLPNetwork, self).__init__()
        layers = []

        if layer_norm:
            layers.append(nn.LayerNorm([in_dim]))
        if batch_norm_first_layer:
            layers.append(nn.BatchNorm1d(in_dim))
        layers += [nn.Linear(in_dim, hidden_dim), unit_activation()]
        for _ in range(num_layers-2):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), unit_activation()]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.Linear(hidden_dim, out_dim), out_activation()]
        self.pipeline = nn.Sequential(*layers)

    def forward(self, x):
        return self.pipeline(x)
