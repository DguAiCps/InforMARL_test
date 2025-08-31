"""
Graph Neural Network components for InforMARL
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EmbedConv(MessagePassing):
    """기존 InforMARL 프로젝트의 EmbedConv 레이어"""
    
    def __init__(self, 
                 input_dim: int = 6,  # [pos, vel, goal] = 6차원
                 num_embeddings: int = 3,  # agent=0, landmark=1, obstacle=2
                 embedding_size: int = 8,
                 hidden_size: int = 64,
                 layer_N: int = 2,
                 edge_dim: int = 1):  # 거리 정보
        super(EmbedConv, self).__init__(aggr='add')
        
        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        
        # 첫 번째 레이어: [node_features + embedding + edge_features] -> hidden
        self.lin1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)
        
        # 은닉 레이어들
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, edge_index, edge_attr=None, entity_type=None):
        """
        x: [num_nodes, input_dim] - 노드 특징
        edge_index: [2, num_edges] - 엣지 연결
        edge_attr: [num_edges, edge_dim] - 엣지 특징 (거리)
        entity_type: [num_nodes] - 엔티티 타입
        """
        # 엔티티 타입 임베딩
        if entity_type is not None:
            entity_emb = self.entity_embed(entity_type.long())
            x = torch.cat([x, entity_emb], dim=-1)
        
        # 메시지 패싱
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr=None):
        """메시지 생성"""
        if edge_attr is not None:
            msg = torch.cat([x_j, edge_attr], dim=-1)
        else:
            msg = x_j
            
        # 첫 번째 레이어
        msg = self.activation(self.lin1(msg))
        msg = self.layer_norm(msg)
        
        # 은닉 레이어들
        for layer in self.layers:
            msg = self.activation(layer(msg))
            msg = self.layer_norm(msg)
            
        return msg


class GraphNeuralNetwork(nn.Module):
    """InforMARL용 GNN - 논문의 제약 조건에 맞게 수정"""
    
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 64, 
                 num_layers: int = 1,  # 관측 범위 제한을 위해 1 layer로 제한
                 num_embeddings: int = 4,  # agent, landmark, obstacle, wall
                 embedding_size: int = 8,
                 edge_dim: int = 1,
                 use_attention: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 첫 번째 (그리고 유일한) 레이어 - 직접 관측 정보만 처리
        self.embed_conv = EmbedConv(
            input_dim=input_dim,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=hidden_dim,
            layer_N=1,
            edge_dim=edge_dim
        )
        
        # Attention 메커니즘 (논문의 selective prioritization)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, data):
        """
        논문의 제약 조건에 맞는 단일 layer 처리
        - 직접 관측 정보만 사용 (1-hop만)
        - Attention을 통한 selective prioritization
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        entity_type = data.entity_type
        
        # 단일 GNN 레이어로 직접 관측 정보만 처리
        x = self.embed_conv(x, edge_index, edge_attr, entity_type)
        
        # Attention 메커니즘 적용 (논문의 selective prioritization)
        if self.use_attention:
            # x를 batch 형태로 변환 [1, num_nodes, hidden_dim]
            x_batch = x.unsqueeze(0)
            
            # Self-attention으로 이웃 노드 정보의 중요도 계산
            attn_output, _ = self.attention(x_batch, x_batch, x_batch)
            
            # Residual connection과 layer normalization
            x = x + attn_output.squeeze(0)
            x = self.attention_norm(x)
        
        # 최종 출력 변환
        x = self.output_layer(x)
        
        return x