# Desc: Test GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0),"Haan bhai gpu hai tere paas")