import torch
import torch.nn as nn

class MILClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3, pooling='attention'):
        super(MILClassifier, self).__init__()

        # Store pooling type
        assert pooling in ['mean', 'max', 'lse', 'attention'], "Pooling must be 'mean', 'max', 'lse', or 'attention'"
        self.pooling = pooling

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)  # Feature extractor (before pooling)

        # Fix: Attention should be applied BEFORE final classification!
        if self.pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),  # Learnable attention mechanism
                nn.Tanh(),
                nn.Linear(128, 1)  # Map to attention score
            )

        # Final classification layer (after pooling)
        self.classifier = nn.Linear(hidden_dims[-1], 1)  # Outputs raw logits (no Sigmoid)

    def forward(self, x, bag_sizes):
        # Compute hidden representations
        instance_features = self.feature_extractor(x)  # Shape (total_instances, hidden_dim)

        # Split predictions into bags
        bag_features = []
        start = 0
        for size in bag_sizes:
            bag = instance_features[start : start + size]  # Extract instances for this bag
            start += size

            if self.pooling == 'mean':
                bag_feature = torch.mean(bag, dim=0)  # Mean pooling
            elif self.pooling == 'max':
                bag_feature = torch.max(bag, dim=0)[0]  # Max pooling
            elif self.pooling == 'lse':  # Log-Sum-Exp pooling
                beta = 5
                bag_feature = torch.logsumexp(beta * bag, dim=0) / beta
            elif self.pooling == 'attention':
                attn_scores = self.attention(bag)  # Compute attention scores (num_instances, 1)
                attn_weights = torch.softmax(attn_scores, dim=0)  # Normalize across instances

                # Compute attention-weighted bag representation
                bag_feature = torch.sum(attn_weights * bag, dim=0)  # Weighted sum of instance features

            bag_features.append(bag_feature)

        # Stack bag features into a single tensor
        bag_features = torch.vstack(bag_features)  # Shape (batch_size, hidden_dim)

        # Final classification layer (logits output, NO SIGMOID applied)
        bag_logits = self.classifier(bag_features)  # Shape (batch_size, 1)

        return bag_logits  # Return logits
