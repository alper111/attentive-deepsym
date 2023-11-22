import math
import os
import torch


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)


class STLayer(torch.nn.Module):

    def __init__(self):
        super(STLayer, self).__init__()
        self.func = StraightThrough.apply

    def forward(self, x):
        return self.func(x)


class STSigmoid(torch.nn.Module):
    def __init__(self):
        super(STSigmoid, self).__init__()

    def forward(self, x):
        m = torch.distributions.Bernoulli(logits=x)
        sample = m.sample()
        probs = torch.sigmoid(x)
        sample = sample + probs - probs.detach()
        return sample


class GumbelSigmoidLayer(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelSigmoidLayer, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        return gumbel_sigmoid(x, self.T, self.hard)


class GumbelSoftmaxLayer(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelSoftmaxLayer, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        return torch.nn.functional.gumbel_softmax(x, tau=self.T, hard=self.hard)


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvTransposeBlock, self).__init__()
        self.block = [torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        o = x.reshape(self.shape)
        return o


class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class ChannelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ChannelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)
        h = self.model(x)
        h = h.reshape(B, -1)
        return h


def sample_gumbel_diff(*shape):
    eps = 1e-20
    u1 = torch.rand(shape)
    u2 = torch.rand(shape)
    diff = torch.log(torch.log(u2+eps)/torch.log(u1+eps)+eps)
    return diff


def gumbel_sigmoid(logits, T=1.0, hard=False):
    g = sample_gumbel_diff(*logits.shape)
    g = g.to(logits.device)
    y = (g + logits) / T
    s = torch.sigmoid(y)
    if hard:
        s_hard = s.round()
        s = (s_hard - s).detach() + s
    return s


class GumbelAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GumbelAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._denom = math.sqrt(out_dim)
        self.wq = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.wk = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x, src_key_mask=None, temperature=1.0, hard=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (batch, token, dim)
        src_key_mask : torch.Tensor
            shape (batch, token)
        temperature : float
            temperature of gumbel sigmoid
        hard : bool
            if True, use hard gumbel sigmoid
        Returns
        -------
        attn : torch.Tensor
            shape (batch, head, token, token)
        """
        if src_key_mask is None:
            src_key_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.float, device=x.device)
        batch, token, dim = x.shape
        x = x.reshape(batch*token, 1, dim, 1)  # (batch*token, placeholder_for_head, in_dim, 1)
        wq = self.wq.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        wk = self.wk.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        pad_mask = src_key_mask.reshape(batch, token, 1, 1)
        q = (wq @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        q = q.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        k = (wk @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        k = k.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        attn = (q @ k.permute(0, 1, 3, 2)) / self._denom  # (batch, head, token, token)
        attn = attn - self.bias
        binarized_attn = gumbel_sigmoid(attn, temperature, hard)
        pad_mask = src_key_mask.reshape(batch, token, 1) @ src_key_mask.reshape(batch, 1, token)
        binarized_attn = binarized_attn * pad_mask.unsqueeze(1)
        return binarized_attn


class GumbelSoftmaxAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GumbelSoftmaxAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._denom = math.sqrt(out_dim)
        self.wq = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.wk = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x, src_key_mask=None, temperature=1.0, hard=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (batch, token, dim)
        src_key_mask : torch.Tensor
            shape (batch, token)
        temperature : float
            temperature of gumbel sigmoid
        hard : bool
            if True, use hard gumbel sigmoid
        Returns
        -------
        attn : torch.Tensor
            shape (batch, head, token, token)
        """
        if src_key_mask is None:
            src_key_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.float, device=x.device)
        batch, token, dim = x.shape
        x = x.reshape(batch*token, 1, dim, 1)  # (batch*token, placeholder_for_head, in_dim, 1)
        wq = self.wq.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        wk = self.wk.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        pad_mask = src_key_mask.reshape(batch, token, 1, 1)
        q = (wq @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        q = q.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        k = (wk @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        k = k.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        attn = (q @ k.permute(0, 1, 3, 2)) / self._denom  # (batch, head, token, token)
        attn = attn - self.bias
        binarized_attn = torch.nn.functional.gumbel_softmax(attn, tau=temperature, hard=hard, axis=-1)
        pad_mask = src_key_mask.reshape(batch, token, 1) @ src_key_mask.reshape(batch, 1, token)
        binarized_attn = binarized_attn * pad_mask.unsqueeze(1)
        return binarized_attn

class STSigmoidAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(STSigmoidAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._denom = math.sqrt(out_dim)
        self.wq = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.wk = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.Tensor(num_heads, out_dim, in_dim)
        ))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.activation = STSigmoid()

    def forward(self, x, src_key_mask=None, temperature=1.0, hard=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (batch, token, dim)
        src_key_mask : torch.Tensor
            shape (batch, token)
        temperature : float
            temperature of gumbel sigmoid
        hard : bool
            if True, use hard gumbel sigmoid
        Returns
        -------
        attn : torch.Tensor
            shape (batch, head, token, token)
        """
        if src_key_mask is None:
            src_key_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.float, device=x.device)
        batch, token, dim = x.shape
        x = x.reshape(batch*token, 1, dim, 1)  # (batch*token, placeholder_for_head, in_dim, 1)
        wq = self.wq.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        wk = self.wk.unsqueeze(0)  # (placeholder_for_batch, head, out_dim, in_dim)
        pad_mask = src_key_mask.reshape(batch, token, 1, 1)
        q = (wq @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        q = q.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        k = (wk @ x).reshape(batch, token, self.num_heads, -1) * pad_mask
        k = k.permute(0, 2, 1, 3)  # (batch, head, token, out_dim)
        attn = (q @ k.permute(0, 1, 3, 2)) / self._denom  # (batch, head, token, token)
        attn = attn - self.bias
        binarized_attn = self.activation(attn)
        pad_mask = src_key_mask.reshape(batch, token, 1) @ src_key_mask.reshape(batch, 1, token)
        binarized_attn = binarized_attn * pad_mask.unsqueeze(1)
        return binarized_attn
