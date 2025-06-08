import torch
import torch.nn as nn

class LSTMModelNoTeacherForcing(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModelNoTeacherForcing, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        seq_length = input.shape[1]
        hidden = None
        outputs = []
        next_input = input[:, 0].unsqueeze(1)

        temp = 0.8
        for _ in range(seq_length):
            embedded = self.embedding(next_input)

            if(hidden == None):
                output, hidden = self.lstm(embedded)
            else:
                output, hidden = self.lstm(embedded, hidden)
            pred = self.fc(output[:, -1, :])

            outputs.append(pred)
            next_input = torch.multinomial(pred.squeeze().div(temp).exp(), 1)

        return outputs[-1]