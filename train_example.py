import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from rae import RAE


def train_batch(model, optimizer, batch):
    output = model(batch)

    optimizer.zero_grad()
    loss = F.mse_loss(output, batch)
    loss.backward()
    optimizer.step()


parser = argparse.ArgumentParser()
parser.add_argument("--input_size", type=int, default=300)
parser.add_argument("--embedding_size", type=int, default=1024)
parser.add_argument("--max_sequence_len", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--cuda", action="store_true")
params, _ = parser.parse_known_args()
params.device = torch.device('cuda:0') if params.cuda else torch.device('cpu')

model = RAE(params)
optimizer = optim.SGD(model.parameters(), lr=params.lr)

batch = torch.arange(900).float().view(1, 3, 300)
train_batch(model, optimizer, batch)
