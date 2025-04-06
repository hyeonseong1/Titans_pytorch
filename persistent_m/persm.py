import torch
import torch.nn as nn

class PersistentMemoryEmbedding(nn.Module):
    """
    Titans 논문 기반 Persistent Memory 모듈

    Args:
        memory_size (int): persistent memory 토큰 수 (Np)
        embed_dim (int): transformer embedding 차원 (d_model)

    example:
        # 입력 설정
        batch_size = 2
        seq_len = 10
        embed_dim = 768
        memory_size = 5

        # 모델 생성
        memory_module = PersistentMemoryEmbedding(memory_size, embed_dim)

        # 가짜 입력
        x = torch.randn(batch_size, seq_len, embed_dim)

        # memory 추가된 입력 생성
        x_aug = memory_module(x)  # (batch_size, seq_len + memory_size, embed_dim)
    other:
        # 각 task마다 메모리를 다르게 관리하려면 딕셔너리로
        memory_dict = {
            "task1": PersistentMemoryEmbedding(memory_size, embed_dim),
            "task2": PersistentMemoryEmbedding(memory_size, embed_dim),
            ...
        }
    """

    def __init__(self, memory_size: int, embed_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))  # (Np, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            x_with_memory: (batch_size, seq_len + memory_size, embed_dim)
        """
        batch_size = x.size(0)

        # 메모리 확장: (batch_size, memory_size, embed_dim)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        # 메모리를 앞에 붙임
        x_with_memory = torch.cat([memory_expanded, x], dim=1)
        return x_with_memory
