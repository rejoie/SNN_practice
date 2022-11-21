import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

lif = neuron.LIFNode(tau=100.)

lif.reset()
x = torch.as_tensor([2.])
T = 150
s_list = []
v_list = []
for t in range(T):
    s_list.append(lif(x))  # 保存每次生成的脉冲
    v_list.append(lif.v)  # 保存每次积分的膜电势

visualizing.plot_one_neuron_v_s(np.asarray(v_list), np.asarray(s_list), v_threshold=lif.v_threshold,
                                v_reset=lif.v_reset, dpi=200)
plt.show()
