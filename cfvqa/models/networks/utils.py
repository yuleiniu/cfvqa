import torch

def mask_softmax(x, lengths):#, dim=1)
    mask = torch.zeros_like(x).to(device=x.device, non_blocking=True)
    t_lengths = lengths[:,:,None].expand_as(mask)
    arange_id = torch.arange(mask.size(1)).to(device=x.device, non_blocking=True)
    arange_id = arange_id[None,:,None].expand_as(mask)

    mask[arange_id<t_lengths] = 1
    # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    # exp(x - max(x)) instead of exp(x) is a trick
    # to improve the numerical stability while giving
    # the same outputs
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class GradReverseMask(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    
    """
    @staticmethod
    def forward(ctx, x, mask, weight):
        """
        The mask should be composed of 0 or 1. 
        The '1' will get their gradient reversed..
        """
        ctx.save_for_backward(mask)
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        mask_c = mask.clone().detach().float()
        mask_c[mask == 0] = 1.0
        mask_c[mask == 1] = - float(ctx.weight)
        return grad_output * mask_c[:, None].float(), None, None


def grad_reverse_mask(x, mask, weight=1):
    return GradReverseMask.apply(x, mask, weight)


class GradReverse(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)



class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)
