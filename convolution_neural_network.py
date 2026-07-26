"""
To do list:
    - donnée de test sur train
    - Accuracy
    
    - a usable version
"""


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import os
import time



# Initie PyTorch pour CPU ou GPU (cuda)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print(f'Torch version : {torch.__version__}')
# print(f'CPU available : {torch.cpu.is_available()}')
print(f'GPU available : {is_cuda}')
print(f'Device : {device}')
if is_cuda:
    print(f'Cuda version : {torch.version.cuda}')
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'GPU properties : {torch.cuda.get_device_properties()}')
    major, minor = torch.cuda.get_device_capability()
    print(f'GPU cuda capability : major = {major}, minor = {minor}')

"""
Localiser les données sois sur CPU ou GPU
Les deux données doivent être au même endroit
"""

# Chargement des images de d'entrainement et de test :
# - 60 000 images d'entrainement
# - 10 000 images de test
base_dir = os.path.dirname(__file__)


# Data d'entrainement : 60 000 images
data_train_path = os.path.join(base_dir, 'dataset.npz')
data_train = np.load(data_train_path)

images_train = data_train['X'] / 255
images_train = torch.from_numpy(images_train).float()

labels_train = data_train['y']
labels_train = torch.from_numpy(labels_train).long()


# Data de test : 10 000 images
data_path_test = os.path.join(base_dir, 'dataset_test.npz')
data_test = np.load(data_path_test)

images_test = data_test['X'] / 255
images_test = torch.from_numpy(images_test).float()

labels_test = data_test['y']
labels_test = torch.from_numpy(labels_test).long()


# Transformation dans le format attendu : (N, C, H, W)
images_train = images_train.view(-1, 1, 28, 28)
images_test = images_test.view(-1, 1, 28, 28)



print('\nData load successfuly...\n')





class Model():
    def __init__(self) -> None:
        self.layers = [
            # === Bloc Convolution 1 ===
            Convolution(out_channels = 32, in_channels = 1, kernel_size = 3, padding = 1),
            ReLU(),
            MaxPooling(),
            # === Bloc Convolution 2 ===
            Convolution(out_channels = 64, in_channels = 32, kernel_size = 3, padding = 1),
            ReLU(),
            MaxPooling(),
            # === Flatten ===
            Flatten(),
            # === Bloc Dense 1 ===
            Dense(in_features = 3136, out_features = 750),
            ReLU(),
            # === Bloc Dense 2 ===
            Dense(in_features = 750, out_features = 150),
            ReLU(),
            # === Bloc Dense 3 ===
            Dense(in_features = 150, out_features = 10)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad) -> None:
        grad = loss_grad

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)



class MaxPooling:
    def forward(self, x):
        N, C, H, W = x.shape
        self.input_shape = x.shape

        x = x.unfold(2, 2, 2).unfold(3, 2, 2)
        # [N, C, H/2, W/2, 2, 2]


        # Chaque pixel du tenseur 2 fois plus petit contient son patch 2x2
        x = x.contiguous().view(N, C, H // 2, W // 2, 4)
        # On aplatie le patch 2x2 en 4
        # .contiguous() -> remet la mémoire en ordre a cause du unfold

        out, self.indices = x.max(dim = -1)
        # On prend le max

        return out

    def backward(self, grad_out):
        N, C, H2, W2 = grad_out.shape
        H, W = self.input_shape[-2:]

        # black vide
        grad_blocks = torch.zeros(
            (N, C, H2, W2, 4),
            device = grad_out.device,
            dtype = grad_out.dtype
        )

        # place le gradient au bon endroit
        grad_blocks.scatter_(
            -1,
            self.indices.unsqueeze(-1),
            grad_out.unsqueeze(-1)
        )

        # 4 -> 2x2
        grad_blocks = grad_blocks.view(N, C, H2, W2, 2, 2)

        # on remet dans l'image
        grad_in = grad_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        grad_in = grad_in.view(N, C, H, W)

        self.indices = None

        return grad_in


class ReLU():
    def forward(self, x):
        self.x = x
        return torch.clamp(x, min = 0)

    def backward(self, grad_out):
        grad_in = grad_out * (self.x > 0)
        self.x = None
        return grad_in


class Flatten():
    def forward(self, x):
        self.input_shape = x.shape
        return x.view(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.view(self.input_shape)


class Dense():
    def __init__(self, in_features: int, out_features: int):
        fan_in = in_features
        self.weights = torch.randn(out_features, in_features) * (2 / fan_in) ** 0.5  # kaiming He
        self.weights = self.weights.to(device)

        self.bias = torch.randn(out_features)
        self.bias = self.bias.to(device)

    def forward(self, x):
        self.x = x
        # (N, in) @ (out, in).t() = (N, out)
        return x @ self.weights.t() + self.bias

    def backward(self, grad_out):
        self.grad_w = grad_out.T @ self.x
        self.grad_b = grad_out.sum(0)
        grad_x = grad_out @ self.weights

        # libération de x pour libérer la VRAM
        self.x = None
        return grad_x

    def step(self, lr):
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b

        self.grad_w = None
        self.grad_b = None


class Convolution():
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int, padding: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # Weight shape : (Cout, Cin, k, k)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * (2 / fan_in) ** 0.5
        self.weight = self.weight.to(device)
        # Bias shape : (Cout,)
        self.bias = torch.zeros(out_channels)
        self.bias = self.bias.to(device)

    def forward(self, x):
        # Ajout du padding
        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        self.x_shape = x.shape

        # Extraction des shapes
        N, C, H, W = self.x_shape
        F, _, kH, kW = self.weight.shape

        # Calcul de la taille de sortie
        H_out = H - kH + 1
        W_out = W - kW + 1

        self.H_out, self.W_out = H_out, W_out

        # Transformer x en patchs (im2col)
        # x_unf shape : [N, C*kH*kW, H_out*W_out]
        self.x_unf = x.unfold(2, kH, 1).unfold(3, kW, 1)
        # Tensor 6D [N, C, kH, kW, H_out, W_out]
        self.x_unf = self.x_unf.contiguous().view(N, C * kH * kW, -1)
        # Aplatit les dimensions du patch (CkHkW) et on combine H_out*W_out
        # Tensor 3D [N, C*kH*kW, H_out*W_out]

        # Reshape des poids : [F, C*kH*kW]
        weight_flat = self.weight.view(F, -1)

        # Multiplication matricielle + ajout du bias
        # out shape : [N, F, H_out*W_out]
        out = torch.matmul(weight_flat, self.x_unf)
        # Produit élément par élément + somme
        out = out + self.bias.view(1, F, 1)

        # Reshape final : [N, F, H_out, W_out]
        out = out.view(N, F, H_out, W_out)

        return out

    def backward(self, grad_out):
        # x : [N, C_in, H, W]
        # w : [C_out, C_in, kH, kW]
        # b : [C_out]
        # out : [N, C_out, H_out, W_out]

        N, C_in, H, W = self.x_shape
        C_out, C_in, kH, kW = self.weight.shape

        H_out, W_out = grad_out.shape[2:]

        w_flat = self.weight.view(C_out, -1)  # [C_out, C_in*kH*kW]

        self.grad_bias = grad_out.sum(dim = (0, 2, 3))  # [C_out]

        grad_out_flat = grad_out.view(N, C_out, H_out*W_out)
        self.grad_weight = grad_out_flat @ self.x_unf.transpose(1, 2)  # [N, C_out, C_in*kH*kW]
        self.grad_weight = self.grad_weight.sum(dim = 0)
        self.grad_weight = self.grad_weight.view(C_out, C_in, kH, kW)

        grad_input_unf = w_flat.t() @ grad_out_flat  # [C_in*kH*kW, H_out*W_out]
        grad_input_unf = grad_input_unf.view(N, C_in, kH, kW, H_out, W_out)
        grad_input_unf = grad_input_unf.permute(0, 1, 4, 2, 5, 3)  # [N ,C_in, H_out, kH, W_out, kW]

        grad_input = torch.zeros(N, C_in, H, W, device = grad_out.device)
        for i in range(kH):
            for j in range(kW):
                grad_input[:, :, i:i + H_out, j:j + W_out] += grad_input_unf[:, :, :, i, :, j]
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        self.x = None
        self.x_unf = None

        return grad_input

    def step(self, lr):
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias

        self.grad_weight = None
        self.grad_bias = None


def train(epochs: int, batch_size: int, lr: float, débuge: bool, test_: bool) -> None:
    epochs = epochs
    # batch de 10000 sur GPU
    batch_size = batch_size
    lr = lr

    nbr_images = epochs * labels_train.shape[0]
    if débuge:
        print(f'{nbr_images} images for training')
        print(f'Estimated time : {nbr_images * 0.000015:.2f}s\n')


    # Création des batchs avec index random
    dataset = TensorDataset(images_train, labels_train)

    print(f'Training strarted...\n')


    # torch.cuda.synchronize()
    stim = time.time()

    i = 0
    sum_loss = 0

    for epoch_nbr in range(epochs):

        loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # forward
            logits = model.forward(x = x)

            N = logits.shape[0]

            # Stabilité
            max_logits = logits.max(dim = 1, keepdim = True).values
            logits = logits - max_logits

            # softmax
            exp = torch.exp(logits)
            probs = exp / exp.sum(dim = 1, keepdim = True)

            # loss
            epsilon = 1e-9
            loss = -torch.log(probs[torch.arange(N), y] + epsilon)
            loss = loss.mean()

            sum_loss += loss

            # gradient
            grad = probs.clone()
            grad[torch.arange(N), y] -= 1
            grad /= N

            # backward
            model.backward(loss_grad = grad)

            model.step(lr = lr)

            i += 1

        if débuge:
            print(f'Loss : {sum_loss / i}')

        if test_:
            if epoch_nbr % (epochs // 10) == 0:
                test()

        i = 0
        sum_loss = 0

    # forward : 60 000 images en 0.58 secondes (GPU)
    # -> 3 000 000 images en 18.32 secondes (vrai test) (GPU)
    # -> 0.00000739 secondes par images (GPU)

    print(f'Training done\n')

    if débuge:
        torch.cuda.synchronize()
        print(f'\nTime : {time.time() - stim:.2f}s')

def test():
    dataset = TensorDataset(images_test, labels_test)
    loader = DataLoader(dataset, batch_size = 1024, shuffle = True)

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # forward
        logits = model.forward(x = x)
        pred = torch.argmax(logits, axis = 1)
        correct += torch.sum(pred == y)
        total += len(x)

    correct = correct.item()

    print('\n', correct, ' / ', total)

# script principal
if __name__ == '__main__':
    model = Model()

    train(
        epochs = 10,
        batch_size = 64,
        lr = 0.01,
        débuge = False,
        test_ = False
    )

    test()

    print(f'\nProgramme executed successfuly')