from torch import nn

def weights_init(tensor):
    classname = tensor.__class__.__name__
    if isinstance(tensor, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        tensor.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(tensor, 'weight') and tensor.weight is not None:
            tensor.weight.data.normal_(1.0, 0.02)
        if hasattr(tensor, 'bias') and tensor.bias is not None:
            tensor.bias.data.fill_(0)

def concat_generators(*args):
    for gen in args:
        yield from gen