from thop import profile

import dist_train
from train import *
import torchvision.datasets as datasets
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_flops_params(model, input_size, device):
    input_sample = torch.randn(1, 3, input_size, input_size).to(device)
    flops, params = profile(model, inputs=(input_sample,))
    return flops, params



if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    input_size = 224
    n_classes = 1000
    model = EfficientViT_M0(num_classes=n_classes, img_size=input_size).to(device)

    flops, params = calculate_flops_params(model, input_size, device)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

