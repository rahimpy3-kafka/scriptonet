import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

class McEwanTherapeuticDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='roberta-base', max_len=512):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = str(row['text'])
        
        # Tokenize using Hugging Face's formatting guidelines
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'det_target': torch.tensor(row['is_therapeutic'], dtype=torch.long),
            'cls_target': torch.tensor(row['mechanism_label'], dtype=torch.long),
            'reg_target': torch.tensor(row['intensity_score'], dtype=torch.float)
        }

def get_dataloader(csv_file, batch_size=16):
    """Factory function to build data pipelines."""
    dataset = McEwanTherapeuticDataset(csv_file=csv_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False
    )
    return dataloader

# Quick test routine if run independently
if __name__ == "__main__":
    # Generate the data file first if it doesn't exist
    try:
        loader = get_dataloader("mcewan_therapeutic_corpus.csv")
        batch = next(iter(loader))
        print("Data Pipeline check passed successfully!")
        print("Input Shapes -> IDs:", batch['input_ids'].shape, "Masks:", batch['attention_mask'].shape)
    except FileNotFoundError:
        print("Please run generate_mock_data.py first to create the data pipeline dependencies.")
