import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import argparse, os, sys,time
from typing import Callable, Union

parser = argparse.ArgumentParser(description='STDP learning')
parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:0', help='device')
parser.add_argument('-b', default=200, type=int, help='batch size')
parser.add_argument('-epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-data-dir', default='/data/tanghao/datasets/', type=str, help='root dir of dataset')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args(args=[])
print(args)

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

train_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_data_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.b,
    shuffle=True,
    drop_last=True,
    num_workers=args.j,
    pin_memory=True
)
test_data_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.b,
    shuffle=False,
    drop_last=False,
    num_workers=args.j,
    pin_memory=True
)

def Possion_Encoder(x):
    return torch.rand_like(x).le(x).to(x)

def f_pre(x, w_min, alpha=0.):
    return (x - w_min) ** alpha

def f_post(x, w_max, alpha=0.):
    return (w_max - x) ** alpha

def trace_update(fc,SpikingNeuron,in_linear ,in_spike: torch.Tensor, out_spike: torch.Tensor, tau_pre: float, tau_post: float):
    if SpikingNeuron.trace_pre is None:
        # print('input:',in_spike)
        SpikingNeuron.trace_pre = torch.zeros_like(in_spike)
        fc.trace_pre = torch.zeros_like(in_spike)

    if SpikingNeuron.trace_post is None:
        SpikingNeuron.trace_post = torch.zeros_like(out_spike)

    if fc.trace is None:
        fc.trace = torch.zeros_like(in_linear)

    # SpikingNeuron.trace_pre = SpikingNeuron.trace_pre - SpikingNeuron.trace_pre / \
    #     tau_pre + in_spike      # shape = [batch_size, N_in]
    # SpikingNeuron.trace_post = SpikingNeuron.trace_post - \
    #     SpikingNeuron.trace_post / tau_post + \
    #     out_spike  # shape = [batch_size, N_out]

    fc.trace = fc.trace - fc.trace / fc.tau + in_linear      # shape = [batch_size, N_in]
    SpikingNeuron.trace_post = SpikingNeuron.trace_post - \
        SpikingNeuron.trace_post / tau_post + \
        out_spike  # shape = [batch_size, N_out]




def stdp_grad(
    weight: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    w_min: float, w_max: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    

    # [batch_size, N_out, N_in] -> [N_out, N_in]
    # 此处对照更新公式，使用unsqueeze添加更新公式中所缺失的一维
    print(trace_pre.shape,trace_post.shape,weight.shape,trace_pre.unsqueeze(1).shape)
    # [200, 784] [200, 10] [10, 784] [200, 1, 784]
    # torch.Size([200, 10]) torch.Size([200, 10]) torch.Size([10, 784]) torch.Size([200, 1, 10])
    # trace_pre shape = [batch_size, N_in] = [200, 784]
    # trace_post shape = [batch_size, N_out] = [200, 10]
    # in_spike shape = [batch_size, N_in] = [200, 784]
    # out_spike shape = [batch_size, N_out] = [200, 10]
    delta_w_pre = -f_pre(weight, w_min) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
    delta_w_post = f_post(weight, w_max) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return delta_w_pre + delta_w_post

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx,in_linear ,input,fc,SpikingNeuron,tau_pre,tau_post, w_min, w_max, f_pre, f_post):
        if SpikingNeuron.v is None:
            SpikingNeuron.v = torch.zeros_like(input)
            SpikingNeuron.s = torch.zeros_like(input)
        SpikingNeuron.v = SpikingNeuron.v + \
            (1.0 - SpikingNeuron.v) / SpikingNeuron.tau * input
        spike = (SpikingNeuron.v >= SpikingNeuron.v_threshold).to(input)
        if SpikingNeuron.soft_reset:
            SpikingNeuron.v = SpikingNeuron.v-SpikingNeuron.v_threshold*spike
        else:
            SpikingNeuron.v = SpikingNeuron.v * \
                (1-spike) + SpikingNeuron.v_reset * spike

        spike = SpikingNeuron.v.ge(
            SpikingNeuron.v_threshold).to(SpikingNeuron.v)
        
        trace_update(fc,SpikingNeuron,in_linear ,input, spike, tau_pre, tau_post)# todo: 需要考察一下trace_pre到底是linuear层的输出还是lif层的输入 fc.trace
        ctx.save_for_backward(fc.weight.data,in_linear, spike,fc.trace,SpikingNeuron.trace_post)#SpikingNeuron.trace_pre

        ctx.w_min = w_min
        ctx.w_max = w_max
        ctx.f_pre = f_pre
        ctx.f_post = f_post


        return spike

    @staticmethod
    def backward(ctx, grad_output):
        weight,input, output,trace_pre,trace_post = ctx.saved_tensors
        # grad_input = grad_output.clone()
        return grad_output * stdp_grad(weight, input, output,trace_pre, trace_post, ctx.w_min, ctx.w_max, ctx.f_pre, ctx.f_post)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,tau=2.):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.tau = tau
        self.trace = None

    def reset(self):
        self.trace = None

class Linear_Spiking(nn.Module):
    def __init__(self, in_feature, out_feature, f_pre, f_post, tau=2.0, tau_pre=2.0, v_threshold=1.0, v_reset=0.0, soft_reset=True, w_min=-1.0, w_max=1.0):
        super(Linear_Spiking, self).__init__()
        self.tau = tau
        self.v = None
        self.s = None
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.soft_reset = soft_reset
        self.trace_pre = None
        self.trace_post = None
        self.w_min = w_min
        self.w_max = w_max
        self.f_pre = f_pre
        self.f_post = f_post
        self.flatten = nn.Flatten()
        self.linear = Linear(in_feature, out_feature, bias=False, tau=tau_pre)
        # self.lif=LIFNeuron(tau=tau, v_threshold=v_threshold, v_reset=v_reset, soft_reset=soft_reset)

    def reset(self):
        self.linear.reset()
        self.v = None
        self.s = None
        self.trace_pre = None
        self.trace_post = None

    def forward(self, x):
        y = self.flatten(x)
        self.linear.weight.data.clamp_(self.w_min, self.w_max)
        in_spike = self.linear(y)
        # print(input.shape)
        # if self.v is None:
        #     self.v = torch.zeros_like(input)
        #     self.s = torch.zeros_like(input)
        # self.v = self.v + (1.0 - self.v) / self.tau * input
        # spike = (self.v >= self.v_threshold).to(input)
        # if self.soft_reset:
        #     self.v = self.v-self.v_threshold*spike
        # else:
        #     self.v = self.v * (1-spike) + self.v_reset * spike
        # (ctx, input,fc,SpikingNeuron,tau_pre,tau_post, w_min, w_max, f_pre, f_post)
        spike = ActFun.apply(y,in_spike, self.linear, self, self.linear.tau,
                             self.tau, self.w_min, self.w_max, self.f_pre, self.f_post)

        return spike


model=Linear_Spiking(784,10,f_pre, f_post)
# if args.opt == 'adam':
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model=model.to(args.device)

acc_max = 0
epoch_max = 0
for epoch in range(args.epochs):
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    model.train()
    for imag, label in train_data_loader:
        optimizer.zero_grad()
        imag = imag.to(args.device)
        label = label.to(args.device)
        label_onehot = F.one_hot(label, 10).float().to(args.device)
        out_fr = 0.
        for t in range(args.T):
            imag_possion = Possion_Encoder(imag)
            out_fr += model(imag_possion)
            # model.net[1].trace_pre, model.net[1].trece_post, grad = stdp_linear_single_step(
            #     model.net[0], imag_possion, out_fr, model.net[1].trace_pre, model.net[1].trace_post, 1, 1, 0, 1, f_pre, f_post)
            # model.net[0].weight.grad += grad

        out_fr = out_fr / args.T

        loss = F.mse_loss(out_fr, label_onehot)
        # model.net[1].trace_pre, model.net[1].trece_post, model.net[0].weight.grad = stdp_linear_single_step(
        #     model.net[0], imag_possion, out_fr, model.net[1].trace_pre, model.net[1].trace_post, 1, 1, 0, 1, f_pre, f_post)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

    print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Time: {:.4f}'.format(
        epoch, train_loss / train_samples, train_acc / train_samples, time.time() - start_time))
