from torch.nn import init
from .correlation_package.correlation import Correlation
from .submodules import *
'Parameter count , 39,175,298 '


class CorrFeature(nn.Module):
    def __init__(self, args, batchNorm=True, div_flow=20):
        super(CorrFeature, self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        if args.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1),
                tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2,
                                    corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3::, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # [N, 441, H, W]
        return out_corr
