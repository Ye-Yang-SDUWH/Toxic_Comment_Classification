import torch
import torch.nn.functional as F
from torch import nn


class Capsule(nn.Module):
    def __init__(self, input_dim, num_capsule=10, dim_capsule=16, routings=3, kernel_size=(9, 1)):
        super().__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.activation = self.squash

        self.W = nn.Parameter(torch.randn((self.num_capsule * self.dim_capsule, input_dim, 1)))

    def forward(self, u_vecs):
        # u_vecs.shape = [N, C, L]
        u_hat_vecs = F.conv1d(u_vecs, self.W).permute([0, 2, 1]) # [N, L, n*d]

        batch_size = u_hat_vecs.shape[0]
        input_num_capsule = u_hat_vecs.shape[1]
        u_hat_vecs = u_hat_vecs.reshape(batch_size, input_num_capsule,
                                        self.num_capsule, self.dim_capsule)
        u_hat_vecs = torch.permute(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = F.softmax(b.permute([0, 2, 1]))
            c = torch.permute(c, (0, 2, 1)).contiguous()  # shape = [None, num_capsule, input_num_capsule]
            outputs = self.activation(torch.einsum('bni, bnid -> bnd', c, u_hat_vecs))
            # shape = [None, num_capsule, dim_capsule]
            if i < self.routings - 1:
                b = torch.einsum('bnd, bnid -> bni', outputs, u_hat_vecs)

        return outputs

    def squash(self, x, axis=-1):
        # s_squared_norm is really small
        # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
        # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
        # return scale * x
        s_squared_norm = torch.sum(x ** 2, dim=axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-10)
        return x / scale
