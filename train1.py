from dataset import get_dataloader
from train import setup_training

# 1. Prepare data stream pipeline
train_loader = get_dataloader("mcewan_therapeutic_corpus.csv", batch_size=16)

# 2. Spin up setup parameters
model, tokenizer, optimizer, criterion, device = setup_training()

# 3. Pull an active training loop
for epoch in range(10):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Multi-task targets
        det_target = batch['det_target'].to(device)
        cls_target = batch['cls_target'].to(device)
        reg_target = batch['reg_target'].to(device)
        
        # Forward execution pass...
