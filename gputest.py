# Desc: Test GPU availability
import torch
print("Haan cuda available hai tere system mein",torch.cuda.is_available())
print(torch.cuda.get_device_name(0)," you have GPU bro :)")
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
print(torch.cuda.get_device_capability())