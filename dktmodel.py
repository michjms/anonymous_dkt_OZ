from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class DKTModel(nn.Module):
    def __init__(self, num_skills, hidden_dim=200, num_layers=1):
        super(DKTModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(num_skills * 2, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class DKTDataset(Dataset):
    def __init__(self, sequences, num_skills):
        self.sequences = sequences
        self.num_skills = num_skills

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        skills = seq[:, 0].astype(int)
        corrects = seq[:, 1].astype(int)

        inputs = np.zeros((len(seq), self.num_skills * 2))
        for i, (s, c) in enumerate(zip(skills, corrects)):
            inputs[i, s + c * self.num_skills] = 1

        targets = np.zeros((len(seq), self.num_skills))
        for i in range(len(seq) - 1):
            targets[i, skills[i+1]] = corrects[i+1]

        return (torch.FloatTensor(inputs[:-1]),
                torch.FloatTensor(targets[:-1]),
                torch.LongTensor(skills[1:]),
                len(inputs) - 1) 

def collate_fn(batch):
    inputs, targets, skills, lengths = zip(*batch)

    inputs_padded = pad_sequence(inputs, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    skills_padded = pad_sequence(skills, batch_first=True, padding_value=-1)
    lengths = torch.LongTensor(lengths)

    return inputs_padded, targets_padded, skills_padded, lengths

def splitting(sequence):
  mean_correct = [seq[:, 1].mean() for seq in sequence]
  bins = pd.qcut(mean_correct, q=4, labels=False, duplicates='drop')
  train_seq, test_seq = train_test_split(sequence, test_size=0.2, random_state=42, stratify=bins)
  return train_seq, test_seq