#  import sys
#  sys.path.append("..")

import torch
import torch.nn as nn
# from torch_stft import STFT

model_info = {
    "exp_name": "clean",
    "model_name": "f_tfcn_64",
    "loss_type": "log_mse",
    "model_param": {
        "enc_dim": 256,
        "channel_dim": 64,
        "feature_dim": 256,
        "kernel": (7,7),
        "layer": 8,
        "stack": 4,
    }
}


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x C x T
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class DepthConv2d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1):
        super(DepthConv2d, self).__init__()

        self.conv2d = nn.Conv2d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, kernel, dilation=dilation, groups=hidden_channel, padding=self.padding)
        self.res_out = nn.Conv2d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv2d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv2d(output)))
        residual = self.res_out(output)
        return residual


class tfcn(nn.Module):
    def __init__(self, model_info):
        super(tfcn, self).__init__()
        
        # hyper parameters
        self.enc_dim = model_info["model_param"]["enc_dim"]
        self.channel_dim = model_info["model_param"]["channel_dim"]
        self.feature_dim = model_info["model_param"]["feature_dim"]
        self.kernel = model_info["model_param"]["kernel"]
        self.layer = model_info["model_param"]["layer"]
        self.stack = model_info["model_param"]["stack"]

        # self.norm = GlobalChannelLayerNorm(self.enc_dim, eps=1e-08)
        self.in_block = nn.Conv2d(1,self.channel_dim,(7,7),1,(3,3))
        self.in_nonlinear = nn.PReLU()
        self.ou_block = nn.Conv2d(self.channel_dim,2,1)
        self.tcn = nn.ModuleList([])
        for s in range(self.stack):
            for i in range(self.layer):
                self.tcn.append(DepthConv2d(self.channel_dim, self.feature_dim, 3, padding=2**i, dilation=2**i))


    def forward(self, input):

        # norm_output = self.norm(input)    # B, L, N
        output = torch.unsqueeze(input,1)
        output = self.in_nonlinear(self.in_block(output))
        for i in range(len(self.tcn)):
            residual = self.tcn[i](output)
            output = output + residual
        output = self.ou_block(output)
        # output = torch.squeeze(output,1)

        return output


if __name__ == "__main__":
    x = torch.rand(2, 256, 125)
    nnet = tfcn(model_info)
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)
    print(x.shape)