"""
2D 병목 시나리오 + InforMARL 구현 - 기존 프로젝트 구조 기반
원형 에이전트, 병목 통로, 상하좌우 이동
InforMARL 논문 및 기존 구현을 정확히 따름
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import random
from collections import deque
import math


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


class InforMARLAgent:
    """InforMARL 에이전트"""
    
    def __init__(self, agent_id: int, obs_dim: int = 6, action_dim: int = 4):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크 초기화
        self.gnn = GraphNeuralNetwork().to(self.device)
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.critic = Critic().to(self.device)
        
        # 옵티마이저 (개별 Actor/Critic만, GNN은 별도로 관리)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            lr=0.03
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
        local_obs = torch.stack([exp['local_obs'] for exp in batch]).to(device)  # 텐서들을 스택
        actions = torch.tensor([exp['action'] for exp in batch]).to(device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch], dtype=torch.float32).to(device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(device)
        values = torch.tensor([exp['value'] for exp in batch], dtype=torch.float32).to(device)
        
        # GAE (Generalized Advantage Estimation)
        advantages = self._compute_gae(rewards, values).to(device)
        returns = (advantages + values).to(device)
        
        # PPO 업데이트
        for _ in range(4):  # PPO 에폭
            # 🔥 공유 GNN으로 그래프 데이터 처리 (배치 단위)
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
                agent_emb = node_embeddings[start_idx + self.agent_id]  # ego agent는 자기 자신
                agent_embeddings.append(agent_emb)
                
                # 전역 집계 (Critic용) - 모든 에이전트 노드들의 평균
                # 첫 N개 노드가 에이전트들이라고 가정
                num_agents = len([1 for exp in batch if exp])  # 배치 내 에이전트 수
                agent_nodes = node_embeddings[start_idx:start_idx + min(num_agents, nodes_per_graph)]
                if len(agent_nodes) > 0:
                    global_agg = agent_nodes.mean(dim=0)
                else:
                    global_agg = node_embeddings[start_idx]
                global_embeddings.append(global_agg)
            
            agent_embeddings = torch.stack(agent_embeddings)
            global_embeddings = torch.stack(global_embeddings)
            
            # 🔥 Actor: 로컬 관측 + GNN 집계 정보
            action_probs = self.actor(local_obs, agent_embeddings)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # 🔥 Critic: 전역 집계 정보로 값 함수 계산
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
            
            # 🔥 역전파 - GNN, Actor, Critic 모두 업데이트
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


@dataclass
class Agent2D:
    """2D 원형 에이전트"""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    target_id: int  # 목표 landmark ID
    max_speed: float
    
    def get_distance_to(self, other_x: float, other_y: float) -> float:
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)
    
    def get_distance_to_agent(self, other: 'Agent2D') -> float:
        return self.get_distance_to(other.x, other.y)


@dataclass  
class Landmark2D:
    """2D 목표 지점"""
    id: int
    x: float
    y: float
    radius: float = 0.5


@dataclass
class Obstacle2D:
    """2D 장애물"""
    id: int
    x: float
    y: float
    radius: float


# 엔티티 타입 매핑 (기존 프로젝트와 동일)
ENTITY_TYPES = {"agent": 0, "landmark": 1, "obstacle": 2}


class BottleneckInforMARLEnv(gym.Env):
    """2D 병목 환경 - InforMARL 기반"""
    
    def __init__(self, 
                 num_agents: int = 6,
                 agent_radius: float = 0.5,
                 corridor_width: float = 20.0,
                 corridor_height: float = 10.0,
                 bottleneck_width: float = 1.2,  # 더 좁게 (이전 1.8)
                 bottleneck_position: float = 10.0,
                 sensing_radius: float = 3.0,
                 max_timesteps: int = 300):
        
        super().__init__()
        
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        self.bottleneck_width = bottleneck_width
        self.bottleneck_position = bottleneck_position
        self.sensing_radius = sensing_radius
        self.max_timesteps = max_timesteps
        
        # 행동 공간: [위, 아래, 왼쪽, 오른쪽]
        self.action_space = spaces.Discrete(4)
        
        self.agents: List[Agent2D] = []
        self.landmarks: List[Landmark2D] = []
        self.obstacles: List[Obstacle2D] = []
        self.informarl_agents: List[InforMARLAgent] = []
        
        # 공유 GNN과 옵티마이저
        self.shared_gnn = None
        self.gnn_optimizer = None
        
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        
        # 렌더링 관련
        self.fig = None
        self.ax = None
        self.render_mode = None
        
    def reset(self) -> List[Data]:
        """환경 리셋"""
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        self.agents = []
        self.landmarks = []
        self.obstacles = []
        self.informarl_agents = []
        
        # 장애물 설정 먼저 정의
        # 에이전트 지름 = agent_radius * 2 = 1.0
        # 장애물 간격을 에이전트 지름의 1/3로 설정하여 촘촘하게
        obstacle_spacing = (self.agent_radius * 2) / 3  # ≈ 0.33
        obstacle_radius = obstacle_spacing / 2  # 장애물 반지름
        wall_margin = obstacle_radius * 3  # 벽에서 충분한 거리 확보
        
        # 목표 지점 생성 (벽 장애물에서 충분히 떨어뜨리기)
        for i in range(self.num_agents):
            if i % 2 == 0:  # L->R
                target_x = np.random.uniform(self.corridor_width - 3.0, self.corridor_width - wall_margin)
            else:  # R->L
                target_x = np.random.uniform(wall_margin, 3.0)
            
            target_y = np.random.uniform(wall_margin, self.corridor_height - wall_margin)
            
            landmark = Landmark2D(id=i, x=target_x, y=target_y)
            self.landmarks.append(landmark)
        
        # 장애물 생성 - 환경의 모든 벽을 장애물 노드로 둘러싸기
        
        obstacle_id = 0
        
        # 1. 상단 경계벽 (전체 너비)
        num_top_obstacles = int(self.corridor_width / obstacle_spacing) + 1
        for i in range(num_top_obstacles):
            x_pos = i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=x_pos,
                y=self.corridor_height - obstacle_radius,  # 상단 경계
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 2. 하단 경계벽 (전체 너비)
        for i in range(num_top_obstacles):
            x_pos = i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=x_pos,
                y=obstacle_radius,  # 하단 경계
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 3. 좌측 경계벽 (전체 높이)
        num_left_obstacles = int(self.corridor_height / obstacle_spacing) + 1
        for i in range(num_left_obstacles):
            y_pos = i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=obstacle_radius,  # 좌측 경계
                y=y_pos,
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 4. 우측 경계벽 (전체 높이)
        for i in range(num_left_obstacles):
            y_pos = i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=self.corridor_width - obstacle_radius,  # 우측 경계
                y=y_pos,
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 5. 병목 구간 벽들 - 통로는 완전히 열어둠
        center_y = self.corridor_height / 2
        wall_length = 2.0
        num_bottleneck_obstacles = int(wall_length / obstacle_spacing) + 1
        
        # 병목 통로의 상하 경계 계산 (여유 공간 추가)
        passage_margin = obstacle_radius * 2  # 통로 주변에 충분한 여유 공간
        passage_top = center_y + self.bottleneck_width/2 + passage_margin
        passage_bottom = center_y - self.bottleneck_width/2 - passage_margin
        
        # 병목 위쪽 벽
        upper_wall_y = center_y + self.bottleneck_width/2 + obstacle_radius
        for i in range(num_bottleneck_obstacles):
            x_pos = self.bottleneck_position - wall_length/2 + i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=x_pos,
                y=upper_wall_y,
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 병목 아래쪽 벽
        lower_wall_y = center_y - self.bottleneck_width/2 - obstacle_radius
        for i in range(num_bottleneck_obstacles):
            x_pos = self.bottleneck_position - wall_length/2 + i * obstacle_spacing
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=x_pos,
                y=lower_wall_y,
                radius=obstacle_radius
            )
            self.obstacles.append(obstacle)
            obstacle_id += 1
        
        # 6. 병목 입구 좌측 벽 (좌측에서 병목까지, 통로 부분은 제외)
        left_wall_x = self.bottleneck_position - wall_length/2 - obstacle_radius
        
        # 하단 경계에서 병목 통로 아래까지
        num_bottom_obstacles = int((passage_bottom - obstacle_radius) / obstacle_spacing)
        for i in range(num_bottom_obstacles):
            y_pos = obstacle_radius + i * obstacle_spacing
            # 통로 여유 공간과 겹치지 않도록 더 엄격한 조건
            if y_pos < passage_bottom - passage_margin:
                obstacle = Obstacle2D(
                    id=obstacle_id,
                    x=left_wall_x,
                    y=y_pos,
                    radius=obstacle_radius
                )
                self.obstacles.append(obstacle)
                obstacle_id += 1
        
        # 병목 통로 위에서 상단 경계까지
        start_y = passage_top + passage_margin  # 여유 공간 추가
        num_top_obstacles = int((self.corridor_height - start_y) / obstacle_spacing)
        for i in range(num_top_obstacles):
            y_pos = start_y + i * obstacle_spacing
            if y_pos < self.corridor_height - obstacle_radius:
                obstacle = Obstacle2D(
                    id=obstacle_id,
                    x=left_wall_x,
                    y=y_pos,
                    radius=obstacle_radius
                )
                self.obstacles.append(obstacle)
                obstacle_id += 1
        
        # 7. 병목 입구 우측 벽 (병목에서 우측까지, 통로 부분은 제외)
        right_wall_x = self.bottleneck_position + wall_length/2 + obstacle_radius
        
        # 하단 경계에서 병목 통로 아래까지 (우측벽)
        for i in range(num_bottom_obstacles):
            y_pos = obstacle_radius + i * obstacle_spacing
            if y_pos < passage_bottom - passage_margin:  # 여유 공간과 겹치지 않도록
                obstacle = Obstacle2D(
                    id=obstacle_id,
                    x=right_wall_x,
                    y=y_pos,
                    radius=obstacle_radius
                )
                self.obstacles.append(obstacle)
                obstacle_id += 1
        
        # 병목 통로 위에서 상단 경계까지 (우측벽)
        for i in range(num_top_obstacles):
            y_pos = start_y + i * obstacle_spacing
            if y_pos < self.corridor_height - obstacle_radius:
                obstacle = Obstacle2D(
                    id=obstacle_id,
                    x=right_wall_x,
                    y=y_pos,
                    radius=obstacle_radius
                )
                self.obstacles.append(obstacle)
                obstacle_id += 1
        
        # 에이전트 생성 (벽 장애물에서 충분히 떨어뜨리기)
        for i in range(self.num_agents):
            if i % 2 == 0:  # L->R
                start_x = np.random.uniform(wall_margin, 3.0)
            else:  # R->L
                start_x = np.random.uniform(self.corridor_width - 3.0, self.corridor_width - wall_margin)
            
            start_y = np.random.uniform(wall_margin, self.corridor_height - wall_margin)
            max_speed = np.random.uniform(1.0, 2.0)
            
            agent = Agent2D(
                id=i, x=start_x, y=start_y, vx=0.0, vy=0.0,
                radius=self.agent_radius, target_id=i, max_speed=max_speed
            )
            self.agents.append(agent)
            
            # InforMARL 에이전트 생성
            informarl_agent = InforMARLAgent(agent_id=i)
            self.informarl_agents.append(informarl_agent)
        
        # 공유 GNN 초기화 (첫 번째 에이전트의 GNN을 공유로 사용)
        if self.shared_gnn is None:
            device = self.informarl_agents[0].device
            self.shared_gnn = GraphNeuralNetwork().to(device)
            self.gnn_optimizer = torch.optim.Adam(self.shared_gnn.parameters(), lr=0.03)
        
        return self._get_graph_observations()
    
    def _get_graph_observations(self) -> List[Data]:
        """InforMARL 방식 그래프 관측 생성"""
        observations = []
        
        # 각 에이전트마다 자신을 기준으로 한 그래프 생성
        for ego_agent in self.agents:
            node_features = []
            entity_types = []
            all_entities = []
            
            # 1. 에이전트 노드들
            for agent in self.agents:
                # 상대 위치/속도/목표 계산 (sensing_radius로 정규화)
                rel_x = (agent.x - ego_agent.x) / self.sensing_radius
                rel_y = (agent.y - ego_agent.y) / self.sensing_radius
                rel_vx = (agent.vx - ego_agent.vx) / agent.max_speed
                rel_vy = (agent.vy - ego_agent.vy) / agent.max_speed
                
                # 목표 위치
                target = self.landmarks[agent.target_id]
                rel_goal_x = (target.x - ego_agent.x) / self.sensing_radius
                rel_goal_y = (target.y - ego_agent.y) / self.sensing_radius
                
                features = [rel_x, rel_y, rel_vx, rel_vy, rel_goal_x, rel_goal_y]
                node_features.append(features)
                entity_types.append(ENTITY_TYPES["agent"])
                all_entities.append(('agent', agent))
            
            # 2. 목표 지점 노드들
            for landmark in self.landmarks:
                rel_x = (landmark.x - ego_agent.x) / self.sensing_radius
                rel_y = (landmark.y - ego_agent.y) / self.sensing_radius
                
                features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # 목표는 정지, 목표=자기위치
                node_features.append(features)
                entity_types.append(ENTITY_TYPES["landmark"])
                all_entities.append(('landmark', landmark))
            
            # 3. 장애물 노드들
            for obstacle in self.obstacles:
                rel_x = (obstacle.x - ego_agent.x) / self.sensing_radius
                rel_y = (obstacle.y - ego_agent.y) / self.sensing_radius
                
                features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # 장애물은 정지
                node_features.append(features)
                entity_types.append(ENTITY_TYPES["obstacle"])
                all_entities.append(('obstacle', obstacle))
            
            # 4. 엣지 생성 (논문의 방향성 규칙에 따라)
            # - non-agent → agent: 단방향 (landmark, obstacle → agent)
            # - agent ↔ agent: 양방향
            edge_index = []
            edge_attr = []
            
            for i, (type_i, entity_i) in enumerate(all_entities):
                for j, (type_j, entity_j) in enumerate(all_entities):
                    if i != j:
                        # ego_agent로부터 두 엔티티까지의 거리 체크
                        if type_i == 'agent':
                            dist_i = ego_agent.get_distance_to_agent(entity_i)
                        else:
                            dist_i = ego_agent.get_distance_to(entity_i.x, entity_i.y)
                            
                        if type_j == 'agent':
                            dist_j = ego_agent.get_distance_to_agent(entity_j)
                        else:
                            dist_j = ego_agent.get_distance_to(entity_j.x, entity_j.y)
                        
                        # 두 엔티티 모두 ego_agent의 센싱 반경 내에 있어야 함
                        if dist_i <= self.sensing_radius and dist_j <= self.sensing_radius:
                            # 엣지 방향성 규칙 적용
                            should_connect = False
                            
                            if type_i == 'agent' and type_j == 'agent':
                                # agent ↔ agent: 양방향 연결
                                should_connect = True
                            elif type_i in ['landmark', 'obstacle'] and type_j == 'agent':
                                # non-agent → agent: 단방향 연결
                                should_connect = True
                            elif type_i == 'agent' and type_j in ['landmark', 'obstacle']:
                                # agent → non-agent: 연결하지 않음 (논문 규칙)
                                should_connect = False
                            else:
                                # non-agent → non-agent: 연결하지 않음
                                should_connect = False
                            
                            if should_connect:
                                edge_index.append([i, j])
                                
                                # 엣지 특징: 두 엔티티 간 거리
                                if type_i == 'agent' and type_j == 'agent':
                                    edge_dist = entity_i.get_distance_to_agent(entity_j)
                                elif type_i == 'agent':
                                    edge_dist = entity_i.get_distance_to(entity_j.x, entity_j.y)
                                elif type_j == 'agent':
                                    edge_dist = entity_j.get_distance_to(entity_i.x, entity_i.y)
                                else:
                                    dx = entity_i.x - entity_j.x
                                    dy = entity_i.y - entity_j.y
                                    edge_dist = math.sqrt(dx*dx + dy*dy)
                                
                                edge_attr.append([edge_dist / self.sensing_radius])  # 정규화
            
            # 최소 연결 보장
            if not edge_index:
                edge_index = [[0, 0]]
                edge_attr = [[0.0]]
            
            # 그래프 데이터 생성
            graph_data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                entity_type=torch.tensor(entity_types, dtype=torch.long)
            )
            
            observations.append(graph_data)
        
        return observations
    
    def step(self, actions: List[int] = None):
        """환경 스텝 실행"""
        self.timestep += 1
        
        # 그래프 관측
        graph_obs = self._get_graph_observations()
        
        # 행동 선택 (배치 처리)
        if actions is None:
            actions, log_probs, values = self._get_batch_actions(graph_obs, training=True)
        else:
            log_probs = [0.0] * len(actions)
            values = [0.0] * len(actions)
        
        # 행동 실행
        for i, action in enumerate(actions):
            self._execute_action(i, action)
        
        # 물리 업데이트
        self._update_positions()
        
        # 보상 계산
        rewards = self._calculate_rewards()
        
        # 경험 저장
        if actions is not None:
            for i, (action, log_prob, value, reward) in enumerate(zip(actions, log_probs, values, rewards)):
                obs = self._get_local_observation(i)
                experience = {
                    'graph_data': graph_obs[i],
                    'local_obs': torch.tensor(obs, dtype=torch.float32),  # 텐서로 변환
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward
                }
                self.informarl_agents[i].store_experience(experience)
        
        new_obs = self._get_graph_observations()
        done = self._is_done()
        info = self._get_info()
        
        return new_obs, rewards, done, info
    
    def _get_batch_actions(self, graph_observations: List[Data], training: bool = True):
        """배치 행동 선택 - InforMARL 방식"""
        device = self.informarl_agents[0].device
        
        # 배치 그래프 생성
        batch_graphs = Batch.from_data_list(graph_observations).to(device)
        
        # 공유 GNN으로 노드 임베딩 계산
        node_embeddings = self.shared_gnn(batch_graphs)
        
        # 에이전트별 정보 추출
        nodes_per_graph = len(graph_observations[0].x)
        num_agents_in_graph = self.num_agents
        
        actions = []
        log_probs = []
        values = []
        
        if training:
            # 학습 시: Actor + Critic 모두 사용
            
            # 각 에이전트의 개별 임베딩 추출 (Actor용)
            agent_embeddings = []
            for i in range(self.num_agents):
                start_idx = i * nodes_per_graph
                # ego_agent는 항상 첫 번째 노드 (인덱스 i)
                agent_emb = node_embeddings[start_idx + i]
                agent_embeddings.append(agent_emb)
            
            # Critic용 전역 집계
            # 각 그래프에서 에이전트 노드들만 추출하여 평균
            graph_embeddings = []
            for i in range(self.num_agents):
                start_idx = i * nodes_per_graph
                # 첫 num_agents개 노드가 에이전트들
                agent_nodes = node_embeddings[start_idx:start_idx + num_agents_in_graph]
                graph_agg = agent_nodes.mean(dim=0)  # 에이전트들의 평균
                graph_embeddings.append(graph_agg)
            
            global_agg = torch.stack(graph_embeddings)  # [num_agents, hidden_dim]
            global_values = self.informarl_agents[0].critic(global_agg)
            
            for i, agent in enumerate(self.informarl_agents):
                # 로컬 관측
                local_obs = torch.tensor(self._get_local_observation(i), dtype=torch.float32).unsqueeze(0).to(device)
                
                # Actor: 로컬 관측 + 집계 정보
                agg_info = agent_embeddings[i].unsqueeze(0)
                action_probs = agent.actor(local_obs, agg_info)
                
                # 확률적 행동 선택
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                actions.append(action.item())
                log_probs.append(log_prob.item())
                values.append(global_values[i].item())
        else:
            # 평가 시: Actor만 사용
            agent_embeddings = []
            for i in range(self.num_agents):
                start_idx = i * nodes_per_graph
                agent_emb = node_embeddings[start_idx + i]
                agent_embeddings.append(agent_emb)
            
            for i, agent in enumerate(self.informarl_agents):
                local_obs = torch.tensor(self._get_local_observation(i), dtype=torch.float32).unsqueeze(0).to(device)
                agg_info = agent_embeddings[i].unsqueeze(0)
                action_probs = agent.actor(local_obs, agg_info)
                
                # 결정적 행동 선택
                action = torch.argmax(action_probs, dim=1)
                actions.append(action.item())
                log_probs.append(0.0)
                values.append(0.0)
        
        return actions, log_probs, values
    
    def _get_local_observation(self, agent_id: int) -> List[float]:
        """에이전트의 로컬 관측 (논문의 o(i))"""
        agent = self.agents[agent_id]
        target = self.landmarks[agent.target_id]
        
        return [
            agent.x / self.sensing_radius,    # sensing_radius로 정규화된 위치
            agent.y / self.sensing_radius,
            agent.vx / agent.max_speed,       # 정규화된 속도
            agent.vy / agent.max_speed,
            (target.x - agent.x) / self.sensing_radius,  # 상대 목표 위치
            (target.y - agent.y) / self.sensing_radius
        ]
    
    def _execute_action(self, agent_id: int, action: int):
        """행동 실행 - 충돌 페널티 적용"""
        agent = self.agents[agent_id]
        
        # 충돌 페널티 체크
        collision_penalty = getattr(agent, 'collision_penalty_timer', 0)
        if collision_penalty > 0:
            speed = agent.max_speed * 0.2  # 충돌 후 느린 속도
            agent.collision_penalty_timer = collision_penalty - 1
        else:
            speed = agent.max_speed * 0.5  # 일반 속도
        
        if action == 0:  # 위
            agent.vy = min(agent.vy + speed, agent.max_speed)
        elif action == 1:  # 아래
            agent.vy = max(agent.vy - speed, -agent.max_speed)
        elif action == 2:  # 왼쪽
            agent.vx = max(agent.vx - speed, -agent.max_speed)
        elif action == 3:  # 오른쪽
            agent.vx = min(agent.vx + speed, agent.max_speed)
        
        # 속도 감쇠 (충돌 페널티 중이면 더 강하게)
        decay_factor = 0.7 if collision_penalty > 0 else 0.9
        agent.vx *= decay_factor
        agent.vy *= decay_factor
    
    def _update_positions(self):
        """위치 업데이트 - 강화된 충돌 처리"""
        dt = 0.1
        
        for agent in self.agents:
            new_x = agent.x + agent.vx * dt
            new_y = agent.y + agent.vy * dt
            
            # 통합된 충돌 체크 (경계, 벽, 장애물, 에이전트 모두 포함)
            collision_info = self._check_collision_detailed(agent.id, new_x, new_y)
            
            if not collision_info['has_collision']:
                agent.x = new_x
                agent.y = new_y
            else:
                # 충돌 시 강화된 처리
                self._handle_collision(agent, collision_info)
                self.collision_count += 1
    
    def _check_collision_detailed(self, agent_id: int, new_x: float, new_y: float) -> Dict[str, Any]:
        """상세 충돌 정보 체크"""
        agent_radius = self.agents[agent_id].radius
        collision_info = {
            'has_collision': False,
            'collision_type': None,
            'collision_entity': None,
            'penetration_depth': 0.0
        }
        
        # 1. 환경 경계 체크
        if (new_x - agent_radius < 0 or new_x + agent_radius > self.corridor_width or
            new_y - agent_radius < 0 or new_y + agent_radius > self.corridor_height):
            collision_info.update({
                'has_collision': True,
                'collision_type': 'boundary',
                'penetration_depth': max(
                    max(0, (agent_radius - new_x)),  # 왼쪽 경계
                    max(0, (new_x + agent_radius - self.corridor_width)),  # 오른쪽 경계
                    max(0, (agent_radius - new_y)),  # 아래 경계
                    max(0, (new_y + agent_radius - self.corridor_height))  # 위 경계
                )
            })
            return collision_info
        
        # 2. 병목 벽 충돌 체크
        if not self._can_pass_through_bottleneck(new_x, new_y, agent_radius):
            collision_info.update({
                'has_collision': True,
                'collision_type': 'bottleneck_wall'
            })
            return collision_info
        
        # 3. 장애물 충돌 체크
        for obstacle in self.obstacles:
            dist = math.sqrt((new_x - obstacle.x)**2 + (new_y - obstacle.y)**2)
            min_dist = agent_radius + obstacle.radius
            if dist < min_dist:
                collision_info.update({
                    'has_collision': True,
                    'collision_type': 'obstacle',
                    'collision_entity': obstacle,
                    'penetration_depth': min_dist - dist
                })
                return collision_info
        
        # 4. 다른 에이전트와의 충돌 체크
        for i, other in enumerate(self.agents):
            if i != agent_id:
                dist = math.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
                min_dist = agent_radius + other.radius
                if dist < min_dist:
                    collision_info.update({
                        'has_collision': True,
                        'collision_type': 'agent',
                        'collision_entity': other,
                        'penetration_depth': min_dist - dist
                    })
                    return collision_info
        
        return collision_info
    
    def _handle_collision(self, agent: Agent2D, collision_info: Dict[str, Any]):
        """강화된 충돌 처리"""
        collision_type = collision_info['collision_type']
        
        # 즉시 속도를 0으로 만들어 추가 침투 방지
        agent.vx = 0.0
        agent.vy = 0.0
        
        # 충돌 타입별 추가 처리
        if collision_type == 'boundary':
            # 경계 충돌: 경계에서 밀어내기
            margin = agent.radius + 0.1
            if agent.x < margin:
                agent.x = margin
            elif agent.x > self.corridor_width - margin:
                agent.x = self.corridor_width - margin
            if agent.y < margin:
                agent.y = margin
            elif agent.y > self.corridor_height - margin:
                agent.y = self.corridor_height - margin
                
        elif collision_type in ['obstacle', 'agent']:
            # 장애물/에이전트 충돌: 충돌 엔티티로부터 밀어내기
            entity = collision_info['collision_entity']
            penetration = collision_info.get('penetration_depth', 0.0)
            
            if penetration > 0.01:  # 충분한 침투가 있을 때만
                # 충돌 방향 계산
                dx = agent.x - entity.x
                dy = agent.y - entity.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0.01:  # 0으로 나누기 방지
                    # 정규화된 방향으로 밀어내기
                    push_distance = penetration + 0.1  # 약간 여유를 둠
                    agent.x += (dx / dist) * push_distance
                    agent.y += (dy / dist) * push_distance
        
        # 충돌한 에이전트는 다음 스텝에서 잠시 느리게 움직임
        agent.collision_penalty_timer = getattr(agent, 'collision_penalty_timer', 0) + 3
    
    
    
    def _can_pass_through_bottleneck(self, x: float, y: float, radius: float) -> bool:
        """병목 구간 통과 가능 여부 체크"""
        center_y = self.corridor_height / 2
        bottleneck_x = self.bottleneck_position
        
        # 병목 구역이 아니면 통과 가능
        if abs(x - bottleneck_x) > 1.0:
            return True
        
        # 병목 구역 내에서는 통로 폭 체크 (에이전트 반지름 고려)
        passage_top = center_y + self.bottleneck_width / 2
        passage_bottom = center_y - self.bottleneck_width / 2
        
        # 에이전트가 통로 안에 완전히 들어갈 수 있는지 체크
        agent_top = y + radius
        agent_bottom = y - radius
        
        return (agent_bottom >= passage_bottom and agent_top <= passage_top)
    
    def _calculate_rewards(self) -> List[float]:
        """개선된 보상 계산 - 좌우 반복 이동 방지"""
        rewards = []
        
        for agent in self.agents:
            reward = 0.0
            target = self.landmarks[agent.target_id]
            distance = agent.get_distance_to(target.x, target.y)
            
            # 목표 도달 - 큰 보상
            if distance < target.radius:
                reward += 100.0
                self.success_count += 1
                rewards.append(reward)
                continue
            
            # 1. 거리 기반 기본 보상 (음수로 시작해서 가까워질수록 덜 나쁨)
            reward -= distance * 0.1
            
            # 2. 목표 방향 이동 보상 (가장 중요!)
            target_direction = np.array([target.x - agent.x, target.y - agent.y])
            target_distance = np.linalg.norm(target_direction)
            
            if target_distance > 0.1:  # 목표에 충분히 멀 때만
                target_direction = target_direction / target_distance  # 정규화
                agent_velocity = np.array([agent.vx, agent.vy])
                velocity_magnitude = np.linalg.norm(agent_velocity)
                
                if velocity_magnitude > 0.05:  # 움직이고 있을 때만
                    agent_direction = agent_velocity / velocity_magnitude
                    # 목표 방향으로의 속도 성분 (내적)
                    direction_alignment = np.dot(agent_direction, target_direction)
                    reward += direction_alignment * velocity_magnitude * 2.0
                else:
                    # 정지해있으면 약간의 페널티
                    reward -= 0.1
            
            # 3. 거리 개선 보상 (이전보다 감소)
            prev_dist = getattr(agent, 'prev_distance', distance)
            if distance < prev_dist:
                improvement = prev_dist - distance
                reward += improvement * 10.0  # 더 큰 보상
            agent.prev_distance = distance
            
            # 4. 반복 움직임 페널티
            prev_positions = getattr(agent, 'position_history', [])
            current_pos = (agent.x, agent.y)
            
            # 최근 위치 기록 (최대 10개)
            prev_positions.append(current_pos)
            if len(prev_positions) > 10:
                prev_positions.pop(0)
            agent.position_history = prev_positions
            
            # 같은 위치 반복 체크
            if len(prev_positions) >= 5:
                recent_positions = prev_positions[-5:]
                position_variance = np.var([pos[0] for pos in recent_positions]) + np.var([pos[1] for pos in recent_positions])
                if position_variance < 0.1:  # 거의 같은 자리
                    reward -= 2.0  # 정체 페널티
            
            # 5. 충돌 페널티 - 충돌 시 큰 마이너스 보상
            collision_penalty = getattr(agent, 'collision_penalty_timer', 0)
            if collision_penalty > 0:
                reward -= 5.0  # 충돌 시 큰 페널티
            
            # 6. 시간 페널티 (너무 오래 걸리면)
            reward -= 0.01  # 매 스텝마다 작은 시간 페널티
            
            rewards.append(reward)
        
        return rewards
    
    def _is_done(self) -> bool:
        """종료 조건"""
        if self.timestep >= self.max_timesteps:
            return True
        
        # 모든 에이전트가 목표 도달
        for agent in self.agents:
            target = self.landmarks[agent.target_id]
            if agent.get_distance_to(target.x, target.y) >= target.radius:
                return False
        return True
    
    def _get_info(self) -> Dict[str, Any]:
        """정보 반환"""
        return {
            "timestep": self.timestep,
            "success_count": self.success_count,
            "collision_count": self.collision_count,
            "success_rate": self.success_count / max(1, self.num_agents),
            "avg_time_ratio": self.timestep / self.max_timesteps
        }
    
    def train_agents(self, num_episodes: int = 100):
        """실제 학습이 포함된 함수"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            observations = self.reset()
            episode_reward = 0
            
            for step in range(self.max_timesteps):
                observations, rewards, done, info = self.step()
                episode_reward += sum(rewards)
                
                # 🔥 매 N스텝마다 네트워크 업데이트
                if step % 10 == 0:
                    for agent in self.informarl_agents:
                        agent.update_networks(self.shared_gnn)
                    # 공유 GNN 옵티마이저 스텝
                    self.gnn_optimizer.step()
                    self.gnn_optimizer.zero_grad()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 🔥 에피소드 끝에 한 번 더 업데이트
            for agent in self.informarl_agents:
                agent.update_networks(self.shared_gnn)
            # 공유 GNN 옵티마이저 스텝
            self.gnn_optimizer.step()
            self.gnn_optimizer.zero_grad()
                
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Success Rate = {info['success_rate']:.2f}")
        
        return episode_rewards
    
    def render(self, mode='human'):
        """환경 렌더링"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            plt.ion()  # 인터랙티브 모드
        
        self.ax.clear()
        
        # 환경 전체 배경
        self.ax.fill_between([0, self.corridor_width], 0, self.corridor_height, 
                            color='lightblue', alpha=0.2, label='복도')
        
        # 병목 구역 표시 (회색 벽들)
        center_y = self.corridor_height / 2
        bottleneck_x = self.bottleneck_position
        
        # 위쪽 벽
        upper_wall = patches.Rectangle(
            (bottleneck_x - 0.5, center_y + self.bottleneck_width/2), 
            1.0, self.corridor_height - (center_y + self.bottleneck_width/2),
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(upper_wall)
        
        # 아래쪽 벽  
        lower_wall = patches.Rectangle(
            (bottleneck_x - 0.5, 0), 
            1.0, center_y - self.bottleneck_width/2,
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(lower_wall)
        
        # 병목 통로 표시 (노란색으로 강조)
        bottleneck_passage = patches.Rectangle(
            (bottleneck_x - 0.5, center_y - self.bottleneck_width/2),
            1.0, self.bottleneck_width,
            facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2
        )
        self.ax.add_patch(bottleneck_passage)
        
        # 환경 경계 테두리
        boundary = patches.Rectangle(
            (0, 0), self.corridor_width, self.corridor_height,
            linewidth=3, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(boundary)
        
        # 병목 장애물 그리기 (원형 장애물들)
        for obstacle in self.obstacles:
            obs_circle = patches.Circle(
                (obstacle.x, obstacle.y), obstacle.radius,
                color='red', alpha=0.9, edgecolor='darkred', linewidth=2
            )
            self.ax.add_patch(obs_circle)
        
        # 목표 지점 그리기
        for i, landmark in enumerate(self.landmarks):
            goal_circle = patches.Circle(
                (landmark.x, landmark.y), landmark.radius,
                color='green', alpha=0.7, edgecolor='darkgreen', linewidth=2
            )
            self.ax.add_patch(goal_circle)
            self.ax.text(landmark.x, landmark.y, f'G{i}', 
                        ha='center', va='center', fontweight='bold', fontsize=10)
        
        # 에이전트 그리기
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for i, agent in enumerate(self.agents):
            color = colors[i % len(colors)]
            
            # 에이전트 원
            agent_circle = patches.Circle(
                (agent.x, agent.y), agent.radius,
                color=color, alpha=0.8, edgecolor='black', linewidth=1
            )
            self.ax.add_patch(agent_circle)
            
            # 에이전트 ID
            self.ax.text(agent.x, agent.y, str(i), 
                        ha='center', va='center', fontweight='bold', 
                        color='white', fontsize=8)
            
            # 목표까지의 선
            target = self.landmarks[agent.target_id]
            self.ax.plot([agent.x, target.x], [agent.y, target.y], 
                        color=color, alpha=0.5, linestyle='--', linewidth=1.5)
            
            # 속도 벡터 (더 명확하게)
            speed = math.sqrt(agent.vx**2 + agent.vy**2)
            if speed > 0.1:
                scale = 3.0  # 화살표 크기 조정
                self.ax.arrow(agent.x, agent.y, agent.vx*scale, agent.vy*scale,
                            head_width=0.15, head_length=0.15, 
                            fc=color, ec=color, alpha=0.8, linewidth=2)
        
        # 설정
        self.ax.set_xlim(-0.5, self.corridor_width + 0.5)
        self.ax.set_ylim(-0.5, self.corridor_height + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'InforMARL 2D Bottleneck - Step {self.timestep}\\n성공: {self.success_count}, 충돌: {self.collision_count}', 
                         fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # 범례 추가
        legend_elements = [
            patches.Patch(color='darkgray', label='벽'),
            patches.Patch(color='yellow', alpha=0.3, label='병목 통로'),
            patches.Patch(color='green', label='목표'),
            patches.Patch(color='red', label='장애물')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # 강제로 화면 업데이트
        plt.draw()
        plt.pause(0.01)
        
        if mode == 'human':
            plt.show(block=False)
    
    def evaluate_with_animation(self, num_episodes: int = 5, render_delay: float = 0.2):
        """평가 모드로 에이전트 실행하며 애니메이션 표시"""
        print("=== InforMARL 평가 모드 (애니메이션) ===")
        print("창이 열리면 에이전트 움직임을 관찰하세요!")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n에피소드 {episode + 1}/{num_episodes} 시작")
            observations = self.reset()
            episode_reward = 0
            
            # 초기 상태 렌더링
            self.render()
            time.sleep(render_delay * 2)  # 초기 상태 좀 더 오래 보여주기
            
            for step in range(self.max_timesteps):
                # 평가 모드로 행동 선택 (training=False)
                actions, _, _ = self._get_batch_actions(
                    self._get_graph_observations(), training=False
                )
                
                # 한 스텝 실행
                observations, rewards, done, info = self.step(actions)
                episode_reward += sum(rewards)
                
                # 렌더링
                self.render()
                
                # 움직임 확인을 위한 디버그 출력
                if step % 20 == 0:
                    print(f"    스텝 {step}: 에이전트 위치들")
                    for i, agent in enumerate(self.agents):
                        print(f"      Agent {i}: ({agent.x:.1f}, {agent.y:.1f}) 속도: ({agent.vx:.2f}, {agent.vy:.2f})")
                
                time.sleep(render_delay)
                
                if done:
                    print(f"  에피소드 완료! 스텝: {step + 1}")
                    break
            
            episode_rewards.append(episode_reward)
            print(f"  에피소드 보상: {episode_reward:.2f}")
            print(f"  성공률: {info['success_rate']:.2f}")
            print(f"  충돌 횟수: {info['collision_count']}")
            
            # 에피소드 간 잠시 대기
            print("  다음 에피소드까지 잠시 대기...")
            time.sleep(2.0)
        
        avg_reward = np.mean(episode_rewards)
        print(f"\n=== 평가 결과 ===")
        print(f"평균 에피소드 보상: {avg_reward:.3f}")
        
        return episode_rewards


def run_informarl_experiment(num_episodes: int = 100, num_agents: int = 6):
    """InforMARL 실험 실행"""
    print("=== InforMARL 2D 병목 환경 학습 시작 ===")
    
    env = BottleneckInforMARLEnv(num_agents=num_agents)
    episode_rewards = env.train_agents(num_episodes=num_episodes)
    
    print(f"\n=== 최종 결과 ===")
    print(f"평균 에피소드 보상: {np.mean(episode_rewards):.3f}")
    
    return episode_rewards, env


def run_animation_demo(num_agents: int = 4):
    """애니메이션 데모 실행"""
    print("=== InforMARL 애니메이션 데모 ===")
    
    env = BottleneckInforMARLEnv(num_agents=num_agents)
    
    # 간단한 학습 (선택사항)
    print("간단한 사전 학습 중...")
    env.train_agents(num_episodes=10)
    
    # 애니메이션으로 평가
    print("\n평가 모드 애니메이션 시작!")
    results = env.evaluate_with_animation(num_episodes=3, render_delay=0.2)
    
    return results


def quick_movement_test(num_agents: int = 2):
    """에이전트 움직임 빠른 테스트"""
    print("=== 에이전트 움직임 테스트 ===")
    
    env = BottleneckInforMARLEnv(num_agents=num_agents, max_timesteps=50)
    observations = env.reset()
    
    print("초기 상태 렌더링...")
    env.render()
    time.sleep(1)
    
    for step in range(20):
        # 랜덤 행동으로 테스트
        random_actions = [np.random.randint(0, 4) for _ in range(num_agents)]
        
        print(f"스텝 {step}: 행동 {random_actions}")
        observations, rewards, done, info = env.step(random_actions)
        
        # 에이전트 위치 출력
        for i, agent in enumerate(env.agents):
            print(f"  Agent {i}: ({agent.x:.2f}, {agent.y:.2f}) 속도: ({agent.vx:.2f}, {agent.vy:.2f})")
        
        env.render()
        time.sleep(0.5)
        
        if done:
            break
    
    print("테스트 완료!")
    plt.show()
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            # 애니메이션 데모 실행
            run_animation_demo(num_agents=4)
        elif sys.argv[1] == "test":
            # 빠른 움직임 테스트
            quick_movement_test(num_agents=2)
        else:
            print("사용법: python bottleneck_informarl_v2.py [demo|test]")
    else:
        # 일반 학습 실행
        results, env = run_informarl_experiment(num_episodes=100, num_agents=4)
        
        # 학습 후 애니메이션 보기
        print("\n학습 완료! 애니메이션으로 결과 확인 (y/n)?")
        if input().lower() == 'y':
            env.evaluate_with_animation(num_episodes=2, render_delay=0.15)