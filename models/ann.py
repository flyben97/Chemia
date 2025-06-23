# train_valid_test/models/ann.py
import torch
import torch.nn as nn

class ComplexANN(nn.Module):
    """
    ANN model for regression or classification.
    Now includes Dropout for regularization.
    """
    # --- MODIFICATION: Add dropout_rate to the constructor ---
    def __init__(self, input_size, hidden_sizes, output_size, task_type='regression', dropout_rate=0.5):
        super(ComplexANN, self).__init__()
        self.task_type = task_type
        
        layers = []
        current_input_size = input_size
        
        if not hidden_sizes:
            # If there are no hidden layers, we don't apply dropout here.
            # The structure is just input -> output.
            pass
        else:
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(current_input_size, hidden_size))
                # It's common to add Batch Normalization here as well for deeper networks
                # layers.append(nn.BatchNorm1d(hidden_size)) 
                layers.append(nn.ReLU())
                # --- MODIFICATION: Add Dropout layer ---
                layers.append(nn.Dropout(p=dropout_rate))
                current_input_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_input_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x