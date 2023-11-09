import torch
import time

# cpu speed vs gpu speed

# cpu
x = torch.randn(10000, 10000)
y = torch.randn(10000, 10000)
start = time.time()
z = x @ y
end = time.time()
print("Cpu: ", end - start)

# gpu
x = torch.randn(10000, 10000).cuda()
y = torch.randn(10000, 10000).cuda()
start = time.time()
z = x @ y
end = time.time()
print("Gpu: ", end - start)