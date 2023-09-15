import torch.nn as nn
import torch
class LIF(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        reset_mechanism="subtract",
    ):
        super().__init__(
        )
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.spike_grad = self.Heaviside.apply
        # self.register_buffer("mem", mem)
    # def init(self):

    def forward(self, cur, mem):
        # arg = torch.zeros_like(cur, requires_grad=True)
        spk = self.spike_grad(cur - self.threshold).clone().detach()
        mem = self.beta * mem + cur- spk*self.threshold
        return spk, mem


    @staticmethod
    class Heaviside(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_):
            out = (input_ > 0).float()
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (out,) = ctx.saved_tensors
            grad = grad_output * out
            return grad
