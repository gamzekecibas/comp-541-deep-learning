import torch.nn as nn

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GraphAttentionNetwork, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.attn_fc = nn.Linear(2 * out_features, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        x = self.fc(x)
        N = x.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        x0 = x.repeat(N, 1, 1)
        x1 = x.repeat(1, N, 1).view(N * N, -1)
        edge = torch.cat([x0, x1], dim=2).view(-1, 2 * self.out_features)
        edge_e = self.leakyrelu(self.attn_fc(edge)).view(-1, N)
        zero_vec = -9e15 * torch.ones_like(edge_e)
        attention = torch.where(adj > 0, edge_e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention.unsqueeze(1), x)

        return h_prime
