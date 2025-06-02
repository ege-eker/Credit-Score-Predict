import torch.nn as nn
import torch

class OrdinalNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Ordinal output layers (binary classifiers for each threshold)
        self.ordinal_layers = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(num_classes - 1)
        ])

    def forward(self, x):
        features = self.features(x)

        # Get ordinal predictions
        ordinal_logits = []
        for layer in self.ordinal_layers:
            ordinal_logits.append(layer(features))

        # Stack and apply sigmoid
        ordinal_logits = torch.cat(ordinal_logits, dim=1)
        ordinal_probs = torch.sigmoid(ordinal_logits)

        # Convert to class predictions
        # Sum up the probabilities to get the predicted class
        predictions = torch.sum(ordinal_probs > 0.5, dim=1)

        return ordinal_logits, predictions
