import torch
use_gpu = torch.cuda.is_available()
torch.cuda.empty_cache()
print(use_gpu)