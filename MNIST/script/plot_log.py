import numpy as np
import matplotlib.pyplot as plt

loss = []
with open("./log/IWAE_hidden_size_50_num_samples_10.log", 'r') as file_handle:
    for line in file_handle:
        line = line.strip()
        fields = line.split(":")
        loss.append(float(fields[-1]))

fig = plt.figure(0)
plt.plot(loss)
plt.ylim(0,150)
plt.show()
