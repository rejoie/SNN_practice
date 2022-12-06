import torch
import torch.nn as nn

#  Check if we have a CUDA-capable device; if so, use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Will train on {}'.format(device))


#  定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3,1),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.net(x)
        return y
    
#  载入模型与输入，并打印此时的模型参数
x = (torch.rand(3)).to(device)
net = CNN().to(device)
print('the first output!')
for name, parameters in net.named_parameters():
    print(name, ':', parameters)
    
#  为了让参数恢复成初始化状态，使用最简单的SGD优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
print('-------------------------------------------------------------------------------')    
#  做梯度下降
optimizer.zero_grad()
y = net(x)
loss = (1-y)**2

loss.backward()
optimizer.step()
#  打印梯度信息
for name, parameters in net.named_parameters():
    print(name, ':', parameters.grad)
#  经过第一次更新以后，打印网络参数
for name, parameters in net.named_parameters():
    print(name, ':', parameters)
    
print('-------------------------------------------------------------------------------')
#  我们直接将网络参数的梯度信息改为相反数来进行梯度上升
for name, parameters in net.named_parameters():
    parameters.grad *= -1
#  打印
for name, parameters in net.named_parameters():
    print('the second output!')
    print(name, ':', parameters.grad)
