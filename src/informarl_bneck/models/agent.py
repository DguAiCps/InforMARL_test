"""
InforMARL Agent implementation with PPO training
"""
import torch
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict

from .policy import Actor, Critic
from ..utils.device import get_device


class InforMARLAgent:
    """InforMARL 에이전트"""
    
    def __init__(self, agent_id: int, obs_dim: int = 6, action_dim: int = 4, device=None):
        self.agent_id = agent_id
        self.device = device if device is not None else get_device()
        
        # 네트워크 초기화 (GNN은 공유되므로 별도)
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.critic = Critic().to(self.device)
        
        # 옵티마이저 (개별 Actor/Critic만, GNN은 별도로 관리)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            lr=0.003
        )
        
        # 경험 버퍼
        self.memory = deque(maxlen=10000)
        
    def store_experience(self, experience: Dict):
        """경험 저장"""
        self.memory.append(experience)
    
    def get_all_params(self):
        """개별 네트워크 파라미터 반환 (GNN 제외)"""
        params = []
        params.extend(list(self.actor.parameters()))
        params.extend(list(self.critic.parameters()))
        return params
    
    def update_networks(self, shared_gnn):
        """PPO 알고리즘으로 네트워크 업데이트 - 공유 GNN, 개별 Actor/Critic"""
        if len(self.memory) < 32:  # 최소 배치 사이즈
            return
        
        device = self.device
        
        # 경험 샘플링
        batch = random.sample(self.memory, min(32, len(self.memory)))
        
        # 배치 데이터 준비
        graph_data_list = [exp['graph_data'] for exp in batch]
        local_obs = torch.stack([exp['local_obs'] for exp in batch]).to(device)
        actions = torch.tensor([exp['action'] for exp in batch]).to(device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch], dtype=torch.float32).to(device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(device)
        values = torch.tensor([exp['value'] for exp in batch], dtype=torch.float32).to(device)
        
        # GAE (Generalized Advantage Estimation)
        advantages = self._compute_gae(rewards, values).to(device)
        returns = (advantages + values).to(device)
        
        # PPO 업데이트
        for _ in range(4):  # PPO 에폭
            # 공유 GNN으로 그래프 데이터 처리 (배치 단위)
            from torch_geometric.data import Batch
            batch_graphs = Batch.from_data_list(graph_data_list).to(device)
            node_embeddings = shared_gnn(batch_graphs)
            
            # 각 샘플의 에이전트 임베딩 추출 (첫 번째 노드가 ego agent)
            nodes_per_graph = len(graph_data_list[0].x)
            agent_embeddings = []
            global_embeddings = []
            
            for i in range(len(batch)):
                start_idx = i * nodes_per_graph
                # ego agent 임베딩 (Actor용)
                agent_emb = node_embeddings[start_idx + self.agent_id]
                agent_embeddings.append(agent_emb)
                
                # 전역 집계 (Critic용) - 모든 에이전트 노드들의 평균
                num_agents = len([1 for exp in batch if exp])
                agent_nodes = node_embeddings[start_idx:start_idx + min(num_agents, nodes_per_graph)]
                if len(agent_nodes) > 0:
                    global_agg = agent_nodes.mean(dim=0)
                else:
                    global_agg = node_embeddings[start_idx]
                global_embeddings.append(global_agg)
            
            agent_embeddings = torch.stack(agent_embeddings)
            global_embeddings = torch.stack(global_embeddings)
            
            # Actor: 로컬 관측 + GNN 집계 정보
            action_probs = self.actor(local_obs, agent_embeddings)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Critic: 전역 집계 정보로 값 함수 계산
            current_values = self.critic(global_embeddings).squeeze()
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss (clipped)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # 역전파 - Actor, Critic 업데이트
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.get_all_params(), 0.5)
            self.optimizer.step()
    
    def _compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t+1] - values[t]
            
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)