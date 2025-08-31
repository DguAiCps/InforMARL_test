"""
Actor-Critic policy networks for InforMARL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """액터 네트워크 - 개별 에이전트 임베딩 사용"""
    
    def __init__(self, 
                 obs_dim: int = 6,  # 로컬 관측 차원
                 agg_dim: int = 64,  # 집계된 정보 차원
                 action_dim: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        
        # [로컬 관측 + 집계 정보] 연결
        input_dim = obs_dim + agg_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, agg_info):
        """
        obs: [batch, obs_dim] - 로컬 관측
        agg_info: [batch, agg_dim] - 집계된 정보
        """
        x = torch.cat([obs, agg_info], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    """크리틱 네트워크 - 전역 집계 정보 사용"""
    
    def __init__(self, agg_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(agg_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, global_agg):
        """
        global_agg: [batch, agg_dim] - 전역 집계 정보
        """
        x = F.relu(self.fc1(global_agg))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value