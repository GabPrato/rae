import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from rae import RAE


def train(dataset, model, optimizer, params):
    for batch in dataset.train_epoch():
        # expected batch of shape: (batch_size, sequence_length, input_embedding_size)
        batch = torch.from_numpy(batch).float()
        if params.cuda:
            batch = batch.cuda()
        
        output = model(batch)

        optimizer.zero_grad()
        loss = F.mse_loss(output, batch)
        loss.backward()
        optimizer.step()

def test(set_type, dataset, model, params):
    loss = 0.
    batch_count = 0
    for batch in dataset.test_epoch(set_type):
        batch = torch.from_numpy(batch).float()
        if params.cuda:
            batch = batch.cuda()

        output = model(batch)
        loss += F.mse_loss(output, batch).item()

        batch_count += 1
        
    print(f'{set_type.capitalize()} loss: {loss / batch_count}')


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--input_size", type=int, default=300, help="Input embedding size")
parser.add_argument("--embedding_size", type=int, default=1024, help="Sequence embedding size")
parser.add_argument("--max_sequence_len", type=int, default=100)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cuda", action="store_true")
params, _ = parser.parse_known_args()
params.device = torch.device('cuda:0') if params.cuda else torch.device('cpu')


model = RAE(params)
if params.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=params.lr)

dataset = Dataset(params)

for i in range(params.epochs):
    train(dataset, model, optimizer, params)
    test('validation', dataset, model, params)

test('test', dataset, model, params)
