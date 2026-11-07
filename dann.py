from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None
class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.01  # default

    def set_lambda(self, val):
        self.lambda_ = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DANN_DKT(nn.Module):
    def __init__(self, num_skills, hidden_dim=200):
        super().__init__()
        self.feature_extractor = nn.LSTM(num_skills * 2, hidden_dim, batch_first=True)
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_skills)
        )

        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def extract_features(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.feature_extractor(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        batch_size = x.size(0)
        last_indices = (lengths - 1).long()
        features = lstm_out[torch.arange(batch_size), last_indices]
        return features
    def forward(self, x, lengths, return_domain=False):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.feature_extractor(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        label_pred = self.label_predictor(lstm_out)
        label_pred = torch.sigmoid(label_pred)

        if return_domain:
            batch_size = x.size(0)
            last_indices = (lengths - 1).long()
            features = lstm_out.mean(dim=1) 

            reversed_features = self.grl(features)
            domain_pred = self.domain_classifier(reversed_features)
            return label_pred, domain_pred

        return label_pred