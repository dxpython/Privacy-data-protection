"""
动态自适应分块同态加密--DAPHE

CKKS
"""

import tenseal as ts
import numpy as np
from typing import Dict, List
import torch


class DynamicHE:
    """
    支持动态参数调整的同态加密系统

    创新点：
      1. 动态模数调整机制：根据数据规模自动调整多项式模数；
      2. 分层密钥管理：为不同网络层分配独立加密方案；
      3. 非线性近似计算：通过低阶多项式拟合实现激活函数加密计算；
      4. 分步处理：支持分块加密/解密以及模拟中间重缩放，降低连续乘法运算对噪声预算的消耗。
    """

    def __init__(self, init_poly_degree=16384, init_scale=2 ** 20, init_plain_modulus=4096):
        self.poly_modulus_degree = init_poly_degree
        self.scale = init_scale
        self.plain_modulus = init_plain_modulus  
        self._refresh_context()
        # 存储各网络层的加密上下文
        self.layer_schemes: Dict[str, ts.Context] = {}

    def _refresh_context(self):
        """根据当前 poly_modulus_degree 生成新的加密上下文"""
        if self.poly_modulus_degree == 16384:
            coeff_mod_bit_sizes = [60, 40, 40, 60]  # 较大 poly_modulus_degree 配置
        elif self.poly_modulus_degree >= 8192:
            coeff_mod_bit_sizes = [60, 40, 40, 40]  # 较小 poly_modulus_degree 配置
        else:
            coeff_mod_bit_sizes = [50, 30, 30, 50]  # 更小的 poly_modulus_degree 配置

        # 创建加密上下文
        try:
            self.context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                plain_modulus=self.plain_modulus  
            )
            self.context.generate_galois_keys()
            self.context.global_scale = self.scale
        except ValueError as e:
            print(f"加密上下文创建失败：{e}")
            raise e

    def _adjust_parameters(self, data_size: int):
        current_slots = self.poly_modulus_degree // 2
        new_degree = 8192 if data_size <= current_slots else 16384
        if new_degree != self.poly_modulus_degree:
            self.poly_modulus_degree = new_degree
            self._refresh_context()

    def encrypt_layer(self, data: np.ndarray, layer_name: str) -> ts.CKKSVector:
        """
        为不同网络层生成独立的加密上下文并加密数据。
        槽位数必须大于等于数据长度
        """
        if layer_name not in self.layer_schemes:
            required_degree = max(8192, 2 * len(data.flatten()))
            if required_degree == 8192:
                coeff_mod_bit_sizes = [50, 30, 30, 50]
            else:
                coeff_mod_bit_sizes = [60, 40, 40, 40, 60]
            layer_ctx = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=required_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            layer_ctx.generate_galois_keys()
            layer_ctx.global_scale = self.scale
            self.layer_schemes[layer_name] = layer_ctx
        return ts.ckks_vector(self.layer_schemes[layer_name], data.flatten())

    def encrypted_relu(self, vector: ts.CKKSVector, degree=1) -> ts.CKKSVector:
        """
        采用低阶（线性）多项式近似 ReLU 激活函数：
        对于小 x，可近似 tanh(x) ≈ x，
        则 ReLU(x) ≈ 0.5 * x * (1 + x)

        为降低乘法链消耗，首先调用 rescale_ciphertext 刷新噪声预算，
        然后再执行激活函数运算。
        """
        # 模拟中间重缩放操作刷新噪声预算
        vector = self.rescale_ciphertext(vector)
        return 0.5 * vector * (1 + vector)

    def encrypt(self, data: np.ndarray) -> ts.CKKSVector:
        """
        加密输入数据,自动调整参数保证数据能放入单个密文
        同时保存原始形状以便解密后恢复。
        如果数据过大，将尝试自动调整 poly_modulus_degree
        """
        self._adjust_parameters(data.size)
        self.original_shape = data.shape
        return ts.ckks_vector(self.context, data.flatten())

    def decrypt(self, vector: ts.CKKSVector) -> np.ndarray:
        """
        解密数据并恢复原始形状
        """
        decrypted = vector.decrypt()
        if hasattr(self, "original_shape"):
            return np.array(decrypted).reshape(self.original_shape)
        else:
            return np.array(decrypted)

    def aggregate_updates(self, encrypted_updates: list) -> ts.CKKSVector:
        """
        同态安全聚合：对加密向量进行逐元素加法，再求平均
        """
        aggregated = encrypted_updates[0].copy()
        for update in encrypted_updates[1:]:
            aggregated += update
        return aggregated * (1.0 / len(encrypted_updates))

    def dynamic_rekeying(self, new_parameters: dict):
        """
        动态密钥更新机制（创新点）

        根据需求调整系数模数和 poly_modulus_degree
        """
        if new_parameters.get("security_level", "low") == "high":
            new_coeffs = [60, 50, 50, 60]
        else:
            new_coeffs = [50, 30, 30, 50]
        sensitivity = new_parameters.get("data_sensitivity", 0.5)
        self.poly_modulus_degree = int(8192 * (1 + sensitivity))
        if self.poly_modulus_degree < 8192:
            self.poly_modulus_degree = 8192
        if self.poly_modulus_degree == 8192:
            coeff_mod_bit_sizes = new_coeffs
        else:
            coeff_mod_bit_sizes = [new_coeffs[0]] + [new_coeffs[1]] * 3 + [new_coeffs[-1]]
        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = self.scale

    # ---------------- 分块加密/解密 以及模拟重缩放 ----------------
    def encrypt_chunked(self, data: np.ndarray) -> List[ts.CKKSVector]:
        """
        如果待加密数据超过当前密文可容纳的槽位数，则按槽位大小分块加密。
        返回一个 ts.CKKSVector 列表，并保存原始数据形状和每块长度以便解密时恢复。
        """
        flat_data = data.flatten()
        slots = self.poly_modulus_degree // 2
        chunks = [flat_data[i:i + slots] for i in range(0, len(flat_data), slots)]
        ciphertexts = [ts.ckks_vector(self.context, chunk.tolist()) for chunk in chunks]
        self._chunked_data_shape = data.shape
        self._chunk_sizes = [len(chunk) for chunk in chunks]
        return ciphertexts

    def decrypt_chunked(self, ciphertexts: List[ts.CKKSVector]) -> np.ndarray:
        """
        解密分块加密的密文列表，并恢复原始数据形状。
        """
        decrypted_chunks = [np.array(ct.decrypt()) for ct in ciphertexts]
        flat_data = np.concatenate(decrypted_chunks)
        return flat_data.reshape(self._chunked_data_shape)

    def rescale_ciphertext(self, vector: ts.CKKSVector) -> ts.CKKSVector:
        """
        模拟中间重缩放操作：通过解密后再重新加密来“刷新”噪声预算。
        注意：这种方法要求在可信环境下操作，不适用于完全不信任的场景。
        """
        decrypted = vector.decrypt()
        return ts.ckks_vector(self.context, decrypted)


class HEEncryptor:
    """
    面向联邦学习的加密处理器，支持对 PyTorch 张量的加密/解密以及模型参数安全聚合。
    """

    def __init__(self, init_ctx=None):
        self.context = init_ctx or DynamicHE().context

    @staticmethod
    def encrypt_tensor(tensor: np.ndarray, ctx: ts.Context) -> ts.CKKSVector:
        return ts.ckks_vector(ctx, tensor.flatten())

    @staticmethod
    def decrypt_tensor(vector: ts.CKKSVector, shape: tuple) -> torch.Tensor:
        return torch.tensor(vector.decrypt()).view(shape)

    def secure_aggregation(self, encrypted_weights: list) -> dict:
        aggregated_weights = {}
        for key in encrypted_weights[0].keys():
            aggregated = encrypted_weights[0][key].copy()
            for update in encrypted_weights[1:]:
                aggregated += update[key]
            aggregated_weights[key] = aggregated * (1.0 / len(encrypted_weights))
        return aggregated_weights
