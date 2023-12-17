from transformers import BertModel, BertTokenizer
import torch 

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
# load
model = BertModel.from_pretrained(model_name)
input_text = "Here is some text to encode"
# tokenizer-> token_id
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
input_ids = torch.tensor([input_ids])

print(input_ids)