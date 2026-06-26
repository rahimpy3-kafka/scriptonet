import torch
import torch.nn as nn
from transformers import RobertaModel

class ScriptoNet(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_size=256, num_lstm_layers=2, dropout_rate=0.3):
        super(ScriptoNet, self).__init__()
        
        # 1. Contextual Encoder (RoBERTa-base) [cite: 113, 159]
        self.roberta = RobertaModel.from_pretrained(model_name)
        roberta_hidden_size = self.roberta.config.hidden_size
        
        # 2. Shared Sequence Representation (BiLSTM) [cite: 114, 159]
        self.bilstm = nn.LSTM(
            input_size=roberta_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # LSTM output size will be hidden_size * 2 because it's bidirectional
        lstm_output_size = hidden_size * 2
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. Task-Specific Heads [cite: 115]
        
        # Head A: Detection (Binary: Therapeutic Event Yes/No) [cite: 116]
        self.detection_head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(lstm_output_size // 2, 1)
            # Note: Sigmoid is typically applied in the loss function (BCEWithLogitsLoss)
        )
        
        # Head B: Mechanism Classification (5 Classes) [cite: 117]
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(lstm_output_size // 2, 5)
            # Note: Softmax is handled by CrossEntropyLoss
        )
        
        # Head C: Intensity Regression (Continuous score 1-5) [cite: 118]
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(lstm_output_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        # Pass through RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_size)
        
        # Pass through BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        
        # Pool the output (using the first token representation, similar to [CLS] pooling)
        # Alternatively, you could use mean pooling over the sequence length
        pooled_output = lstm_output[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        
        # Task Outputs
        detection_logits = self.detection_head(pooled_output)
        classification_logits = self.classification_head(pooled_output)
        intensity_score = self.regression_head(pooled_output)
        
        return detection_logits, classification_logits, intensity_score
