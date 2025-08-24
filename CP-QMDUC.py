"""TensorCircuit Quantum Convolutional Neural Network with JAX backend and PyTorch batching."""
import os
import random
from typing import Tuple, List, Any, Union
import logging
import pandas as pd
from numpy.ma.core import arcsin
logging.getLogger('jax._src.lib').setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import tensorcircuit as tc
import jax
import jax.numpy as jnp
if not hasattr(jax, 'tree_map'):
    if hasattr(jax, 'tree'):
        jax.tree_map = jax.tree.map
    else:
        jax.tree_map = jax.tree_util.tree_map
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
tc.set_dtype("complex128")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantumSafeNormalize:

    def __init__(self):
        pass

    def __call__(self, tensor):

        tensor_clamped = torch.clamp(tensor, -3.0, 3.0)
        tensor_normalized = 2.0 * torch.sigmoid(tensor_clamped * 0.5) - 1.0
        tensor_safe = torch.clamp(tensor_normalized, -0.99, 0.99)
        tensor_angles = torch.arcsin(tensor_safe)
        if torch.isnan(tensor_angles).any() or torch.isinf(tensor_angles).any():
            print("Warning: NaN or Inf detected in quantum safe normalization!")
            tensor_angles = torch.nan_to_num(tensor_angles, nan=0.0, posinf=1.0, neginf=-1.0)

        return tensor_angles


def load_cifar10_data(root="../data", batch_size=128,
                      train_samples_per_class=640, test_samples_per_class=100,
                      use_quantum_norm=True, image_size=10):

    base_transforms_train = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


    base_transforms_test = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    if use_quantum_norm:
        base_transforms_train.append(QuantumSafeNormalize())
        base_transforms_test.append(QuantumSafeNormalize())
        print(f"使用量子安全归一化预处理，双三次插值缩放至: {image_size}x{image_size}")
    else:
        print(f"使用标准归一化预处理，双三次插值缩放至: {image_size}x{image_size}")

    transform_train = transforms.Compose(base_transforms_train)
    transform_test = transforms.Compose(base_transforms_test)

    full_train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    full_test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    train_labels = np.array(full_train_dataset.targets)
    class_indices_train = {i: np.where(train_labels == i)[0] for i in range(10)}
    selected_train_indices = []
    for i in range(10):
        indices = class_indices_train[i]
        selected = np.random.choice(indices, train_samples_per_class, replace=False)
        selected_train_indices.extend(selected)
    train_dataset = Subset(full_train_dataset, selected_train_indices)

    test_labels = np.array(full_test_dataset.targets)
    class_indices_test = {i: np.where(test_labels == i)[0] for i in range(10)}
    selected_test_indices = []
    for i in range(10):
        indices = class_indices_test[i]
        selected = np.random.choice(indices, test_samples_per_class, replace=False)
        selected_test_indices.extend(selected)
    test_dataset = Subset(full_test_dataset, selected_test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=4)

    print(f"训练集样本数: {len(train_dataset)} ({train_samples_per_class}/类)")
    print(f"测试集样本数: {len(test_dataset)} ({test_samples_per_class}/类)")
    print(f"批次大小: 训练集={batch_size}, 测试集={test_samples_per_class * 10}")
    print(f"插值方法: 双三次插值 (BICUBIC) + 抗锯齿")

    return train_loader, test_loader

cifar10_labels = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

class ChannelWiseQuantumCircuit(nn.Module):


    def __init__(self, in_channels: int, kernel_size: int, layers_per_position: int = 1):

        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_positions = kernel_size * kernel_size
        self.layers_per_position = layers_per_position

        self.weights_per_layer = in_channels * 3

        self.total_weights = self.num_positions * layers_per_position * self.weights_per_layer

        self.weights = nn.Parameter(torch.Tensor(self.total_weights))
        nn.init.uniform_(self.weights, -torch.pi, torch.pi)

    def get_all_weights(self):
        return self.weights

    def execute_circuit(self, weights, multi_channel_patch):

        c = tc.Circuit(self.in_channels)

        for i in range(self.in_channels):
            c.h(i)

        weight_idx = 0

        for pos_idx in range(self.num_positions):
            for channel_idx in range(self.in_channels):
                data_value = multi_channel_patch[channel_idx, pos_idx]
                c.ry(channel_idx, theta=data_value)

            for layer_idx in range(self.layers_per_position):
                for qubit_idx in range(self.in_channels):
                    c.rx(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1

                    c.ry(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1
                    c.rz(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1
                for qubit_idx in range(self.in_channels - 1):
                    c.cx(qubit_idx, qubit_idx + 1)
                if self.in_channels > 1:
                    c.cx(self.in_channels - 1, 0)
        expectations = []
        for i in range(self.in_channels):
            exp_val = c.expectation((tc.gates.z(), [i]))
            exp_val = K.real(exp_val)
            expectations.append(exp_val)

        return jnp.array(expectations)


class ChannelWiseQuantumKernel(nn.Module):


    def __init__(
            self,
            kernel_size: int,
            in_channels: int,
            num_kernels: int = 1,
            layers_per_position: int = 1,
            stride: int = 1,
            dilation: int = 1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.layers_per_position = layers_per_position
        self.stride = stride
        self.dilation = dilation
        self.num_qubits = in_channels

        self.quantum_circuits = nn.ModuleList()
        for _ in range(num_kernels):
            circuit = ChannelWiseQuantumCircuit(
                in_channels=in_channels,
                kernel_size=kernel_size,
                layers_per_position=layers_per_position
            )
            self.quantum_circuits.append(circuit)

        self.quantum_funcs = []
        for i in range(num_kernels):
            quantum_func = tc.interfaces.torch_interface(
                K.vmap(self.quantum_circuits[i].execute_circuit, vectorized_argnums=1),
                jit=True
            )
            self.quantum_funcs.append(quantum_func)

    def extract_multi_channel_patches(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, channels, height, width = x.shape

        out_height = (height - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        out_width = (width - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        num_patches = out_height * out_width

        patches = torch.zeros(
            (batch_size, num_patches, channels, self.kernel_size * self.kernel_size),
            device=x.device
        )

        patch_idx = 0
        for i in range(0, height - (self.kernel_size - 1) * self.dilation, self.stride):
            for j in range(0, width - (self.kernel_size - 1) * self.dilation, self.stride):
                for batch in range(batch_size):
                    for channel in range(channels):
                        flat_idx = 0
                        # 提取当前通道的kernel_size×kernel_size补丁
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                h_idx = i + ki * self.dilation
                                w_idx = j + kj * self.dilation
                                if h_idx < height and w_idx < width:
                                    patches[batch, patch_idx, channel, flat_idx] = x[batch, channel, h_idx, w_idx]
                                flat_idx += 1
                patch_idx += 1

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, channels, height, width = x.shape

        out_height = (height - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        out_width = (width - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1

        patches = self.extract_multi_channel_patches(x)
        # patches shape: [batch_size, num_patches, in_channels, kernel_size²]

        kernel_outputs = []

        for kernel_idx in range(self.num_kernels):
            kernel_output = torch.zeros(
                (batch_size, self.in_channels, out_height, out_width),  # 输出通道数 = 量子比特数 = 输入通道数
                device=x.device
            )

            for batch_idx in range(batch_size):
                batch_patches = patches[batch_idx]  # [num_patches, in_channels, kernel_size²]

                if torch.isnan(batch_patches).any() or torch.isinf(batch_patches).any():
                    print(f"Warning: NaN/Inf in batch {batch_idx}, kernel {kernel_idx}")
                    batch_patches = torch.nan_to_num(batch_patches, nan=0.0)

                try:
                    quantum_results = self.quantum_funcs[kernel_idx](
                        self.quantum_circuits[kernel_idx].get_all_weights(),
                        batch_patches
                    )

                    if torch.isnan(quantum_results).any() or torch.isinf(quantum_results).any():
                        print(f"Warning: NaN/Inf in quantum results for batch {batch_idx}, kernel {kernel_idx}")
                        quantum_results = torch.nan_to_num(quantum_results, nan=0.0)

                    for patch_idx in range(quantum_results.shape[0]):
                        h_idx = patch_idx // out_width
                        w_idx = patch_idx % out_width
                        for channel_idx in range(self.in_channels):
                            kernel_output[batch_idx, channel_idx, h_idx, w_idx] = quantum_results[patch_idx, channel_idx]

                except Exception as e:
                    print(f"Error in quantum computation: {e}")
                    kernel_output[batch_idx] = 0.0

            kernel_outputs.append(kernel_output)

        output = torch.cat(kernel_outputs, dim=1)

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf in final output, applying nan_to_num")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return output

    def get_output_channels(self):
        return self.num_kernels * self.in_channels



class CP_QMDUC_Conv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            num_kernels: int = None,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            layers_per_position: int = 1,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel_size[0] == kernel_size[1], "目前只支持正方形卷积核"

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.kernel_size = kernel_size[0]
        self.layers_per_position = layers_per_position
        self.stride = stride[0]
        self.padding = padding
        self.dilation = dilation[0]

        if num_kernels is not None and out_channels is not None:
            expected_out_channels = num_kernels * in_channels  # 输出通道数 = 卷积核数 × 输入通道数
            assert out_channels == expected_out_channels, \
                f"out_channels({out_channels}) 与 num_kernels * in_channels({expected_out_channels}) 不匹配"
            self.num_kernels = num_kernels
        elif out_channels is not None:
            assert out_channels % in_channels == 0, \
                f"out_channels({out_channels}) 必须能被 in_channels({in_channels}) 整除"
            self.num_kernels = out_channels // in_channels
        elif num_kernels is not None:
            self.num_kernels = num_kernels
        else:
            self.num_kernels = 1

        self.out_channels = self.num_kernels * in_channels

        self.quantum_kernel = ChannelWiseQuantumKernel(
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            num_kernels=self.num_kernels,
            layers_per_position=layers_per_position,
            stride=self.stride,
            dilation=self.dilation,
        )

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x):

        if self.padding[0] > 0 or self.padding[1] > 0:
            x = nn.functional.pad(
                x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            )

        output = self.quantum_kernel(x)

        if self.use_bias:
            output += self.bias.view(1, -1, 1, 1)

        return output

    def get_config_info(self):
        """返回配置信息"""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'num_kernels': self.num_kernels,
            'kernel_size': self.kernel_size,
            'layers_per_position': self.layers_per_position,
            'quantum_qubits': self.in_channels,
            'total_quantum_weights_per_kernel': self.quantum_kernel.quantum_circuits[0].total_weights,
        }


class CP_QMDUC(nn.Module):

    def __init__(self):
        super().__init__()

        KERNEL_SIZE = 2
        LAYERS_PER_POSITION = 1
        NUM_KERNELS = 1

        self.quantum_conv = CP_QMDUC_Conv2d(
            in_channels=3,
            num_kernels=NUM_KERNELS,
            kernel_size=KERNEL_SIZE,
            layers_per_position=LAYERS_PER_POSITION,
            stride=1,
            padding=1,
        )

        quantum_out_channels = self.quantum_conv.out_channels  # = NUM_KERNELS * 3 = 6
        print(f"通道优先量子卷积层配置: {self.quantum_conv.get_config_info()}")

        self.classical_nn = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(quantum_out_channels),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(quantum_out_channels * 5 * 5, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        x = self.quantum_conv(x)
        #x = torch.arcsin(x)/torch.pi
        return self.classical_nn(x)


def save_history_to_shared_csv(history, model_name, filename="cifar_CP_history.csv"):

    data = {
        'model': model_name,
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    }

    df_new = pd.DataFrame(data)

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        if model_name in df_existing['model'].values:
            df_existing = df_existing[df_existing['model'] != model_name]

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(filename, index=False)
    print(f"Training history for {model_name} saved to {filename}")


def train_model(model_name="CIFAR_CP_result(1k1d)"):

    train_loader, test_loader = load_cifar10_data()
    model = CP_QMDUC().to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    #
    # print(f"模型总参数数量: {total_params:,}")
    # print(f"模型可训练参数数量: {trainable_params:,}")
    # print(f"模型不可训练参数数量: {non_trainable_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = f"{model_name}_checkpoint.pt"
    start_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"Loaded checkpoint, resuming from epoch {start_epoch}")
    best_accuracy = 0.0
    if history['val_acc'] and max(history['val_acc']) > best_accuracy:
        best_accuracy = max(history['val_acc'])

    for epoch in range(start_epoch, 50):
        model.train()
        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        train_batches = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                # 累计损失
                train_loss_sum += loss.item()
                train_batches += 1

                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                tepoch.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{train_correct / train_total:.2%}"
                )

        epoch_train_loss = train_loss_sum / train_batches
        epoch_train_acc = train_correct / train_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        test_correct = 0
        test_total = len(test_loader.dataset)
        test_loss_sum = 0.0
        test_batches = 0

        with torch.no_grad(), tqdm(test_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = criterion(output, target)
                test_loss_sum += loss.item()
                test_batches += 1

                test_correct += output.argmax(1).eq(target).sum().item()
                tepoch.set_postfix(acc=f"{test_correct / test_total:.2%}")

        epoch_val_loss = test_loss_sum / test_batches
        epoch_val_acc = test_correct / test_total

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_acc)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_val_loss,
            'history': history
        }, checkpoint_path)

        save_history_to_shared_csv(history, model_name)

        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'history': history
            }, f"{model_name}_best.pt")
            print(f"New best model saved with accuracy: {best_accuracy:.2%}")

        print(f"Epoch {epoch + 1:2d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2%}")

    return history



if __name__ == "__main__":
    train_model()
