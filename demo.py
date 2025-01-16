import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Должно вернуть количество доступных GPU
print(torch.cuda.current_device())  # Должен вернуть индекс текущего устройства
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Должно вывести название вашего GPU
