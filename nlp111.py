
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('uitnlp/visobert', use_fast=True)
print(tok.tokenize('Tôi rất vui 😊 hôm nay 🥰'))
