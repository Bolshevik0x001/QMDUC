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
import torchvision.models.mobilenetv2
# 启用JAX的64位精度
jax.config.update("jax_enable_x64", True)
# 设置TensorCircuit后端为JAX
K = tc.set_backend("jax")
tc.set_dtype("complex128")

def set_seed(seed=42):
    """设置所有随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# 使用示例
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantumSafeNormalize:
    """
    量子安全归一化变换，确保数据适合量子电路处理
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        """
        对输入张量进行量子安全归一化

        Args:
            tensor: 输入张量，通常是经过ToTensor()和标准Normalize后的图像数据

        Returns:
            安全归一化后的张量，适合量子电路处理
        """
        # 先将数据clamp到合理范围
        tensor_clamped = torch.clamp(tensor, -3.0, 3.0)

        # 使用sigmoid函数将数据映射到(0,1)，然后转换到(-1,1)
        tensor_normalized = 2.0 * torch.sigmoid(tensor_clamped * 0.5) - 1.0

        # 进一步确保在arcsin的定义域内
        tensor_safe = torch.clamp(tensor_normalized, -0.99, 0.99)

        # 安全的arcsin变换
        tensor_angles = torch.arcsin(tensor_safe)

        # 数值检查和修复
        if torch.isnan(tensor_angles).any() or torch.isinf(tensor_angles).any():
            print("Warning: NaN or Inf detected in quantum safe normalization!")
            tensor_angles = torch.nan_to_num(tensor_angles, nan=0.0, posinf=1.0, neginf=-1.0)

        return tensor_angles


def load_cifar10_data(root="./data", batch_size=128,
                      train_samples_per_class=5000, test_samples_per_class=1000,
                      use_quantum_norm=True, image_size=10):
    """
    加载CIFAR-10数据集，按类别平衡采样，使用双三次插值进行图片缩放

    Args:
        root: 数据存储路径
        batch_size: 批次大小
        train_samples_per_class: 每个类别的训练样本数量
        test_samples_per_class: 每个类别的测试样本数量
        use_quantum_norm: 是否使用量子安全归一化
        image_size: 目标图片尺寸 (image_size x image_size)

    Returns:
        train_loader, test_loader: 数据加载器
    """

    # 基础变换列表 - 训练集
    base_transforms_train = [
        # 使用双三次插值进行高质量缩放
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True  # 开启抗锯齿，进一步提高质量
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10标准化
    ]

    # 基础变换列表 - 测试集
    base_transforms_test = [
        # 使用双三次插值进行高质量缩放
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True  # 开启抗锯齿
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    # 根据参数决定是否添加量子安全归一化
    if use_quantum_norm:
        base_transforms_train.append(QuantumSafeNormalize())
        base_transforms_test.append(QuantumSafeNormalize())
        print(f"使用量子安全归一化预处理，双三次插值缩放至: {image_size}x{image_size}")
    else:
        print(f"使用标准归一化预处理，双三次插值缩放至: {image_size}x{image_size}")

    # 组合变换
    transform_train = transforms.Compose(base_transforms_train)
    transform_test = transforms.Compose(base_transforms_test)

    # 加载完整CIFAR-10数据集
    full_train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    full_test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    # 按类别平衡采样训练集
    train_labels = np.array(full_train_dataset.targets)
    class_indices_train = {i: np.where(train_labels == i)[0] for i in range(10)}
    selected_train_indices = []
    for i in range(10):
        indices = class_indices_train[i]
        # 确保不重复采样
        selected = np.random.choice(indices, train_samples_per_class, replace=False)
        selected_train_indices.extend(selected)
    train_dataset = Subset(full_train_dataset, selected_train_indices)

    # 按类别平衡采样测试集
    test_labels = np.array(full_test_dataset.targets)
    class_indices_test = {i: np.where(test_labels == i)[0] for i in range(10)}
    selected_test_indices = []
    for i in range(10):
        indices = class_indices_test[i]
        # 确保不重复采样
        selected = np.random.choice(indices, test_samples_per_class, replace=False)
        selected_test_indices.extend(selected)
    test_dataset = Subset(full_test_dataset, selected_test_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    # 测试集使用整个测试集作为单个批次（按原始函数逻辑）
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=4)

    print(f"训练集样本数: {len(train_dataset)} ({train_samples_per_class}/类)")
    print(f"测试集样本数: {len(test_dataset)} ({test_samples_per_class}/类)")
    print(f"批次大小: 训练集={batch_size}, 测试集={test_samples_per_class * 10}")
    print(f"插值方法: 双三次插值 (BICUBIC) + 抗锯齿")

    return train_loader, test_loader

# CIFAR-10类别标签
cifar10_labels = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

class MultiChannelSequentialCircuit(nn.Module):

    def __init__(self, num_qubits: int, in_channels: int, layers_per_channel: int = 1):
        """
        初始化多通道顺序编码量子电路

        Args:
            num_qubits: 量子比特数量
            in_channels: 输入通道数
            layers_per_channel: 每个通道编码后的参数化电路层数
        """
        super().__init__()

        self.num_qubits = num_qubits
        self.in_channels = in_channels
        self.layers_per_channel = layers_per_channel

        # 计算每层参数化电路的权重数量
        self.weights_per_layer = num_qubits * 3  # 每个量子比特3个旋转门

        # 计算总权重数量：每个通道后有layers_per_channel层参数化电路
        self.total_weights = in_channels * layers_per_channel * self.weights_per_layer

        # 创建权重参数
        self.weights = nn.Parameter(torch.Tensor(self.total_weights))
        nn.init.uniform_(self.weights, -torch.pi, torch.pi)

    def get_all_weights(self):
        """获取所有权重参数"""
        return self.weights

    def execute_circuit(self, weights, multi_channel_patch):
        """
        执行多通道顺序编码量子电路

        Args:
            weights: 所有权重参数 [total_weights]
            multi_channel_patch: 多通道补丁数据 [in_channels, kernel_size²]

        Returns:
            期望值 [num_qubits]
        """
        # 创建量子电路
        c = tc.Circuit(self.num_qubits)

        # 初始化：对所有量子比特应用Hadamard门
        for i in range(self.num_qubits):
            c.h(i)

        weight_idx = 0

        # 按通道顺序编码和处理
        for channel_idx in range(self.in_channels):
            # 1. 编码当前通道的数据
            channel_data = multi_channel_patch[channel_idx]  # [kernel_size²]

            # 角度编码：将数据编码到RY旋转门
            for qubit_idx in range(self.num_qubits):
                c.ry(qubit_idx, theta=channel_data[qubit_idx])

            # 2. 应用当前通道对应的参数化量子电路层
            for layer_idx in range(self.layers_per_channel):
                # 应用参数化旋转门
                for qubit_idx in range(self.num_qubits):
                    # RX门
                    c.rx(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1
                    # RY门
                    c.ry(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1
                    # RZ门
                    c.rz(qubit_idx, theta=weights[weight_idx])
                    weight_idx += 1

                # 应用纠缠门
                for qubit_idx in range(self.num_qubits - 1):
                    c.cx(qubit_idx, qubit_idx + 1)
                # 闭合环
                if self.num_qubits > 1:
                    c.cx(self.num_qubits - 1, 0)

        # 3. 测量所有量子比特的Z期望值
        expectations = []
        for i in range(self.num_qubits):
            exp_val = c.expectation((tc.gates.z(), [i]))
            exp_val = K.real(exp_val)
            expectations.append(exp_val)

        return jnp.array(expectations)


class MultiChannelSequentialQuantumKernel(nn.Module):

    def __init__(
            self,
            kernel_size: int,
            in_channels: int,
            num_kernels: int = 1,  # 并行的量子卷积核数量
            layers_per_channel: int = 1,  # 每个通道编码后的参数化电路层数
            stride: int = 1,
            dilation: int = 1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.layers_per_channel = layers_per_channel
        self.stride = stride
        self.dilation = dilation

        # 量子比特数 = kernel_size²
        self.num_qubits = kernel_size * kernel_size

        # 为每个并行的量子卷积核创建独立的电路模块
        self.quantum_circuits = nn.ModuleList()
        for _ in range(num_kernels):
            circuit = MultiChannelSequentialCircuit(
                num_qubits=self.num_qubits,
                in_channels=in_channels,
                layers_per_channel=layers_per_channel
            )
            self.quantum_circuits.append(circuit)

        # 为每个量子卷积核创建可微分的量子函数
        self.quantum_funcs = []
        for i in range(num_kernels):
            quantum_func = tc.interfaces.torch_interface(
                K.vmap(self.quantum_circuits[i].execute_circuit, vectorized_argnums=1),
                jit=True
            )
            self.quantum_funcs.append(quantum_func)

    def extract_multi_channel_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        从多通道输入中提取卷积补丁

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            patches: [batch_size, num_patches, in_channels, kernel_size²]
        """
        batch_size, channels, height, width = x.shape

        # 计算输出特征图尺寸
        out_height = (height - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        out_width = (width - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        num_patches = out_height * out_width

        # 初始化补丁张量
        patches = torch.zeros(
            (batch_size, num_patches, channels, self.kernel_size * self.kernel_size),
            device=x.device
        )

        # 提取每个卷积位置的多通道补丁
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
                                if h_idx < height and w_idx < width:  # 边界检查
                                    patches[batch, patch_idx, channel, flat_idx] = x[batch, channel, h_idx, w_idx]
                                flat_idx += 1
                patch_idx += 1

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            output: [batch_size, num_kernels * num_qubits, out_height, out_width]
        """
        batch_size, channels, height, width = x.shape

        # 计算输出尺寸
        out_height = (height - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        out_width = (width - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1

        # 提取多通道补丁
        patches = self.extract_multi_channel_patches(x)
        # patches shape: [batch_size, num_patches, in_channels, kernel_size²]

        # 存储所有量子卷积核的输出
        kernel_outputs = []

        # 对每个量子卷积核并行处理
        for kernel_idx in range(self.num_kernels):
            kernel_output = torch.zeros(
                (batch_size, self.num_qubits, out_height, out_width),
                device=x.device
            )

            # 处理每个批次
            for batch_idx in range(batch_size):
                batch_patches = patches[batch_idx]  # [num_patches, in_channels, kernel_size²]

                # 数值检查
                if torch.isnan(batch_patches).any() or torch.isinf(batch_patches).any():
                    print(f"Warning: NaN/Inf in batch {batch_idx}, kernel {kernel_idx}")
                    batch_patches = torch.nan_to_num(batch_patches, nan=0.0)

                try:
                    # 量子计算：顺序编码多通道数据
                    quantum_results = self.quantum_funcs[kernel_idx](
                        self.quantum_circuits[kernel_idx].get_all_weights(),
                        batch_patches
                    )

                    # 检查量子计算结果
                    if torch.isnan(quantum_results).any() or torch.isinf(quantum_results).any():
                        print(f"Warning: NaN/Inf in quantum results for batch {batch_idx}, kernel {kernel_idx}")
                        quantum_results = torch.nan_to_num(quantum_results, nan=0.0)

                    # 将结果填入输出张量
                    for patch_idx in range(quantum_results.shape[0]):
                        h_idx = patch_idx // out_width
                        w_idx = patch_idx % out_width
                        for qubit_idx in range(self.num_qubits):
                            kernel_output[batch_idx, qubit_idx, h_idx, w_idx] = quantum_results[patch_idx, qubit_idx]

                except Exception as e:
                    print(f"Error in quantum computation: {e}")
                    kernel_output[batch_idx] = 0.0

            kernel_outputs.append(kernel_output)

        # 连接所有量子卷积核的输出
        output = torch.cat(kernel_outputs, dim=1)

        # 最终数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf in final output, applying nan_to_num")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return output

    def get_output_channels(self):
        """返回输出通道数"""
        return self.num_kernels * self.num_qubits


class EnhancedQuantumConv2d(nn.Module):
    """
    增强版量子卷积层，支持多通道顺序编码 - 修正版
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            num_kernels: int = None,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            layers_per_channel: int = 1,  # 新增：每通道编码后的参数化电路层数
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            bias: bool = True,
    ):
        super().__init__()

        # 处理参数
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
        self.layers_per_channel = layers_per_channel
        self.stride = stride[0]
        self.padding = padding
        self.dilation = dilation[0]

        # 计算量子卷积核数量
        if num_kernels is not None and out_channels is not None:
            expected_out_channels = num_kernels * (self.kernel_size ** 2)
            assert out_channels == expected_out_channels, \
                f"out_channels({out_channels}) 与 num_kernels * kernel_size²({expected_out_channels}) 不匹配"
            self.num_kernels = num_kernels
        elif out_channels is not None:
            kernel_size_sq = self.kernel_size ** 2
            assert out_channels % kernel_size_sq == 0, \
                f"out_channels({out_channels}) 必须能被 kernel_size²({kernel_size_sq}) 整除"
            self.num_kernels = out_channels // kernel_size_sq
        elif num_kernels is not None:
            self.num_kernels = num_kernels
        else:
            self.num_kernels = 1

        self.out_channels = self.num_kernels * (self.kernel_size ** 2)

        # 创建多通道顺序编码量子卷积核
        self.quantum_kernel = MultiChannelSequentialQuantumKernel(
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            num_kernels=self.num_kernels,
            layers_per_channel=layers_per_channel,
            stride=self.stride,
            dilation=self.dilation,
        )

        # 添加偏置
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """前向传播"""
        # 应用填充
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = nn.functional.pad(
                x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            )

        # 量子卷积
        output = self.quantum_kernel(x)

        # 添加偏置
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
            'layers_per_channel': self.layers_per_channel,
            'total_quantum_weights_per_kernel': self.quantum_kernel.quantum_circuits[0].total_weights,
        }


# 使用示例的增强版QCNN
class EnhancedQCNN(nn.Module):
    """
    使用多通道顺序编码的增强版QCNN - 修正版
    """

    def __init__(self):
        super().__init__()

        # 超参数配置
        KERNEL_SIZE = 2
        LAYERS_PER_CHANNEL = 1  # 每个通道编码后应用2层参数化电路
        NUM_KERNELS = 1  # 3个并行的量子卷积核

        # 多通道顺序编码量子卷积层
        self.quantum_conv = EnhancedQuantumConv2d(
            in_channels=3,  # RGB三通道
            num_kernels=NUM_KERNELS,
            kernel_size=KERNEL_SIZE,
            layers_per_channel=LAYERS_PER_CHANNEL,
            stride=1,
            padding=1,
        )

        quantum_out_channels = self.quantum_conv.out_channels
        print(f"增强版量子卷积层配置: {self.quantum_conv.get_config_info()}")

        # 经典神经网络部分
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
        # total_params = sum(p.numel() for p in self.classical_nn.parameters())
        # print(total_params)
        # # x = torch.arcsin(x) / torch.pi
        return self.classical_nn(x)


def save_history_to_shared_csv(history, model_name, filename="cifar_HDQCNN_M_history.csv"):
    """
    将训练历史追加到共享CSV文件

    Args:
        history: 包含训练历史的字典
        model_name: 模型名称
        filename: CSV文件名
    """

    # 准备数据
    data = {
        'model': model_name,
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    }

    df_new = pd.DataFrame(data)

    # 检查文件是否已存在
    if os.path.exists(filename):
        # 读取现有文件
        df_existing = pd.read_csv(filename)

        # 检查该模型是否已存在于文件中
        if model_name in df_existing['model'].values:
            # 删除该模型的旧记录
            df_existing = df_existing[df_existing['model'] != model_name]

        # 合并新旧数据
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # 保存到CSV
    df_combined.to_csv(filename, index=False)
    print(f"Training history for {model_name} saved to {filename}")


def train_model(model_name="HDQCNN_M_CIFAR_result(1k1d)full"):
    """训练模型并记录训练过程中的指标"""
    # 数据加载
    train_loader, test_loader = load_cifar10_data()

    # 模型初始化
    model = EnhancedQCNN().to(device)

    # # 获取模型的可训练参数数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    #
    # print(f"模型总参数数量: {total_params:,}")
    # print(f"模型可训练参数数量: {trainable_params:,}")
    # print(f"模型不可训练参数数量: {non_trainable_params:,}")

    # 添加权重正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = f"{model_name}_checkpoint.pt"
    start_epoch = 0

    # 创建字典来存储训练指标
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    # 检查是否有检查点可以恢复
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"Loaded checkpoint, resuming from epoch {start_epoch}")

    # 记录最佳准确率
    best_accuracy = 0.0
    if history['val_acc'] and max(history['val_acc']) > best_accuracy:
        best_accuracy = max(history['val_acc'])

    for epoch in range(start_epoch, 50):
        # 训练阶段
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

                # 计算训练准确率
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 更新进度条
                tepoch.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{train_correct / train_total:.2%}"
                )

        # 计算平均训练损失和准确率
        epoch_train_loss = train_loss_sum / train_batches
        epoch_train_acc = train_correct / train_total

        # 记录训练指标
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # 验证阶段
        model.eval()
        test_correct = 0
        test_total = len(test_loader.dataset)
        test_loss_sum = 0.0
        test_batches = 0

        with torch.no_grad(), tqdm(test_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # 计算验证损失
                loss = criterion(output, target)
                test_loss_sum += loss.item()
                test_batches += 1

                # 计算验证准确率
                test_correct += output.argmax(1).eq(target).sum().item()
                tepoch.set_postfix(acc=f"{test_correct / test_total:.2%}")

        # 计算平均验证损失和准确率
        epoch_val_loss = test_loss_sum / test_batches
        epoch_val_acc = test_correct / test_total

        # 记录验证指标
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # 学习率调度
        scheduler.step(epoch_val_acc)

        # 保存检查点，包括训练历史
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_val_loss,
            'history': history
        }, checkpoint_path)

        # 每个周期后保存训练历史到CSV文件
        save_history_to_shared_csv(history, model_name)

        # 保存最佳模型
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

        # 打印周期统计
        print(f"Epoch {epoch + 1:2d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2%}")

    return history



if __name__ == "__main__":
    train_model()
