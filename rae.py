import torch
import torch.nn as nn


class RAE(nn.Module):
    def __init__(self, params):
        super(RAE, self).__init__()

        self.params = params
        self.timestep_encoding_size = (params.max_sequence_len + 10) // 3

        hidden_layer_size = params.embedding_size - ((params.embedding_size - params.input_size) // 2)
        
        self.scale_embedding = nn.Sequential(
            nn.Linear(params.input_size, hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, params.embedding_size),
            nn.ReLU(inplace=True)
        )

        self.descale_embedding = nn.Sequential(
            nn.Linear(params.embedding_size, hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, params.input_size)
        )

        hidden_layer_size = round(1.5 * params.embedding_size)

        self.encoder = nn.Sequential(
            nn.Linear(2 * params.embedding_size + self.timestep_encoding_size, hidden_layer_size),
            nn.LayerNorm(hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, params.embedding_size),
            nn.LayerNorm(params.embedding_size),
            nn.ReLU(inplace=True)
        )

        # Decoder is split into two parts because overlapping sets needs to be meaned before applying final LayerNorm and ReLU
        self.decoder_part1 = nn.Sequential(
            nn.Linear(params.embedding_size + self.timestep_encoding_size, hidden_layer_size),
            nn.LayerNorm(hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, 2 * params.embedding_size)
        )
        self.decoder_part2 = nn.Sequential(
            nn.LayerNorm(params.embedding_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.scale_embedding(x)
        x, recursion_count = self.encode(x)
        x = self.decode(x, recursion_count)
        x = self.descale_embedding(x)
        return x

    def encode(self, x):
        for recursion_count in range(1, x.shape[1]):
            x = torch.cat((x[:, :-1], x[:, 1:]), 2)
            timestep_encoding = self.get_timestep_encoding(x, recursion_count)
            x = torch.cat((x, timestep_encoding), dim=2)
            x = self.encoder(x)
        return x, recursion_count

    def decode(self, x, recursion_count):
        timestep_encoding = self.get_timestep_encoding(x, recursion_count)
        x = torch.cat((x, timestep_encoding), dim=2)
        x = self.decoder_part1(x)
        x = x.view(x.shape[0], 2, self.params.embedding_size)
        x = self.decoder_part2(x)

        for recursion_count in range(recursion_count - 1, 0, -1):
            timestep_encoding = self.get_timestep_encoding(x, recursion_count)
            x = torch.cat((x, timestep_encoding), dim=2)
            x = self.decoder_part1(x)
            
            overlap = torch.mul(x[:, 0:-1, self.params.embedding_size:] + x[:, 1:, :self.params.embedding_size], 0.5)
            x = torch.cat((x[:, 0:1, :self.params.embedding_size], overlap, x[:, -1:, self.params.embedding_size:]), dim=1)
            x = self.decoder_part2(x)

        return x

    def get_timestep_encoding(self, x, recursion_count):
        encoding = torch.zeros((x.shape[0], x.shape[1], self.timestep_encoding_size), device=self.params.device)
        encoding[:, :, 0] = recursion_count
        if recursion_count == 1:
            encoding[:, :, 1] = 1
        elif recursion_count == 2:
            encoding[:, :, 2] = 1
        elif recursion_count >= 3 and recursion_count <= 4:
            encoding[:, :, 3] = 1
        else:
            encoding[:, :, ((recursion_count + 10) // 3) - 1] = 1
            
        return encoding