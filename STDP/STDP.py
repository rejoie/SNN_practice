import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import argparse
import os
import sys
import time
from typing import Callable, Union

from torch.nn.init import kaiming_uniform_
import math


parser = argparse.ArgumentParser(description='STDP learning')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:0', help='device')
parser.add_argument('-b', default=512, type=int, help='batch size')
parser.add_argument('-epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-data-dir', default='/data/tanghao/datasets/',
                    type=str, help='root dir of dataset')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'],
                    default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9,
                    type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-2, type=float, help='learning rate')
# parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

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


def f_pre(x, w_min, alpha=1.):
    return (x - w_min) ** alpha


def f_post(x, w_max, alpha=1.):
    return (w_max - x) ** alpha


def trace_update(SpikingNeuron, in_spike: torch.Tensor, out_spike: torch.Tensor, tau_pre: float, tau_post: float):
    if SpikingNeuron.trace_pre is None:
        # print('input:',in_spike)
        SpikingNeuron.trace_pre = torch.zeros_like(in_spike)

    if SpikingNeuron.trace_post is None:
        SpikingNeuron.trace_post = torch.zeros_like(out_spike)

    # SpikingNeuron.trace_pre = SpikingNeuron.trace_pre - SpikingNeuron.trace_pre / \
    #     tau_pre + in_spike      # shape = [batch_size, N_in]
    # SpikingNeuron.trace_post = SpikingNeuron.trace_post - \
    #     SpikingNeuron.trace_post / tau_post + \
    #     out_spike  # shape = [batch_size, N_out]

    SpikingNeuron.trace_pre = SpikingNeuron.trace_pre - SpikingNeuron.trace_pre / \
        tau_pre + in_spike      # shape = [batch_size, N_in]
    SpikingNeuron.trace_post = SpikingNeuron.trace_post - \
        SpikingNeuron.trace_post / tau_post + \
        out_spike  # shape = [batch_size, N_out]


def stdp_grad(
    weight, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    w_min: float, w_max: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):  # : nn.Linear

    # [batch_size, N_out, N_in] -> [N_out, N_in]
    # 此处对照更新公式，使用unsqueeze添加更新公式中所缺失的一维
    # print(trace_pre.shape,trace_post.shape,weight.shape,trace_pre.unsqueeze(1).shape)
    # [200, 784] [200, 10] [10, 784] [200, 1, 784]
    # torch.Size([200, 784]) torch.Size([200, 10]) torch.Size([10, 784]) torch.Size([200, 1, 784])
    # trace_pre shape = [batch_size, N_in] = [200, 784]
    # trace_post shape = [batch_size, N_out] = [200, 10]
    # in_spike shape = [batch_size, N_in] = [200, 784]
    # out_spike shape = [batch_size, N_out] = [200, 10]
    delta_w_pre = -f_pre(weight, w_min) * \
        (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
    delta_w_post = f_post(weight, w_max) * \
        (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return delta_w_pre + delta_w_post


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, SpikingNeuron, tau_pre, tau_post, w_min, w_max, f_pre, f_post):
        in_spike = input.mm(weight.t())
        if SpikingNeuron.v is None:
            SpikingNeuron.v = torch.zeros_like(in_spike)
            SpikingNeuron.s = torch.zeros_like(in_spike)

        SpikingNeuron.v = SpikingNeuron.v + \
            (-SpikingNeuron.v+in_spike) / tau_post  # todo: 此处的tau
        spike = SpikingNeuron.v.ge(
            SpikingNeuron.v_threshold).to(SpikingNeuron.v)
        # SpikingNeuron.v = SpikingNeuron.v + \
        #     (1.0 - SpikingNeuron.s) / tau_post * in_spike# todo: 此处的tau
        # spike = (SpikingNeuron.v >= SpikingNeuron.v_threshold).to(in_spike)
        if SpikingNeuron.soft_reset:
            SpikingNeuron.v = SpikingNeuron.v-SpikingNeuron.v_threshold*spike
        else:
            SpikingNeuron.v = SpikingNeuron.v * \
                (1-spike) + SpikingNeuron.v_reset * spike

        # todo: 需要考察一下trace_pre到底是linuear层的输出还是lif层的输入 fc.trace
        trace_update(SpikingNeuron, input, spike, tau_pre, tau_post)
        # ctx.save_for_backward(fc.weight.data,input, spike,SpikingNeuron.trace_pre,SpikingNeuron.trace_post)
        ctx.save_for_backward(input, weight)
        ctx.output = spike
        ctx.trace_pre = SpikingNeuron.trace_pre
        ctx.trace_post = SpikingNeuron.trace_post

        ctx.w_min = w_min
        ctx.w_max = w_max
        ctx.f_pre = f_pre
        ctx.f_post = f_post

        # fc.weight.grad = stdp_grad(fc.weight.data, input, spike,SpikingNeuron.trace_pre,SpikingNeuron.trace_post, w_min, w_max, f_pre, f_post)

        return spike

    @staticmethod
    def backward(ctx, grad_output):
        # weight,input, output,trace_pre,trace_post = ctx.saved_tensors
        input, weight = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_output.shape=[200, 10], stdp_grad.shape=[10, 784]
        # print(grad_output.shape, stdp_grad(weight, input, output,trace_pre, trace_post, ctx.w_min, ctx.w_max, ctx.f_pre, ctx.f_post).shape)
        # 此处返回梯度维度应该为[200, 784]，与输入维度一致
        # return torch.mm(grad_output,weight+stdp_grad(weight, input, ctx.output,ctx.trace_pre, ctx.trace_post, ctx.w_min, ctx.w_max, ctx.f_pre, ctx.f_post)),None,None,None,None,None,None,None,None#grad_output.mm(weight)
        # grad_output.mm(weight)
        return grad_output.mm(weight), grad_output.t().mm(input)+stdp_grad(weight, input, ctx.output, ctx.trace_pre, ctx.trace_post, ctx.w_min, ctx.w_max, ctx.f_pre, ctx.f_post), None, None, None, None, None, None, None
        # return torch.ones_like(input)
        # return grad_output


class Linear_Spiking(nn.Module):
    def __init__(self, in_features, out_features, f_pre, f_post, tau_pre=2.0, tau_post=2.0, v_threshold=0.5, v_reset=0.0, soft_reset=True, w_min=-1.0, w_max=1.0):
        super(Linear_Spiking, self).__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
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
        # self.linear = nn.Linear(in_feature, out_feature, bias=False)
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.lif=LIFNeuron(tau=tau, v_threshold=v_threshold, v_reset=v_reset, soft_reset=soft_reset)

    def reset(self):
        self.v = None
        self.s = None
        self.trace_pre = None
        self.trace_post = None

    def forward(self, input):
        self.weight.data.clamp_(self.w_min, self.w_max)
        # (ctx,input,fc,SpikingNeuron,tau_pre,tau_post, w_min, w_max, f_pre, f_post)
        spike = ActFun.apply(input, self.weight, self, self.tau_pre,
                             self.tau_post, self.w_min, self.w_max, self.f_pre, self.f_post)
        # spike=(torch.rand(200,10)>0.7).to(args.device)

        return spike


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 200),
    Linear_Spiking(200, 10, f_pre, f_post)
)
# if args.opt == 'adam':
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model = model.to(args.device)

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
            # imag_possion = encoder(imag)
            # model[1].weight.data.clamp_(0, 1)
            out_fr += model(imag_possion)
            # model.net[1].trace_pre, model.net[1].trece_post, grad = stdp_linear_single_step(
            #     model.net[0], imag_possion, out_fr, model.net[1].trace_pre, model.net[1].trace_post, 1, 1, 0, 1, f_pre, f_post)
            # model.net[0].weight.grad += grad

        out_fr = out_fr / args.T

        # out_fr.requires_grad = True

        loss = F.mse_loss(out_fr, label_onehot)
        # model.net[1].trace_pre, model.net[1].trece_post, model.net[0].weight.grad = stdp_linear_single_step(
        #     model.net[0], imag_possion, out_fr, model.net[1].trace_pre, model.net[1].trace_post, 1, 1, 0, 1, f_pre, f_post)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        model[2].reset()

    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for imag, label in test_data_loader:
            imag = imag.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float().to(args.device)
            out_fr = 0.
            for t in range(args.T):
                imag_possion = Possion_Encoder(imag)
                # imag_possion = encoder(imag)
                out_fr += model(imag_possion)
            out_fr = out_fr / args.T
            loss = F.mse_loss(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            model[2].reset()

    print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}s'.format(
        epoch, train_loss / train_samples, test_loss/test_samples, train_acc / train_samples, test_acc/test_samples, time.time() - start_time))
