"""
Main bottleneck environment for InforMARL
"""
import gymnasium as gym
from gymnasium import spaces
import torch
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any
import numpy as np

from ..utils.types import Agent2D, Landmark2D, Obstacle2D
from ..models import GraphNeuralNetwork, InforMARLAgent
from ..utils.device import get_device, setup_gpu_environment
from .map import create_agents_and_landmarks, create_obstacles
from .physics import execute_action, update_positions, batch_execute_actions_gpu, batch_update_positions_gpu
from .reward import calculate_rewards
from .waypoint_reward import calculate_waypoint_rewards, calculate_waypoint_rewards_gpu
from .graph_builder import build_graph_observations, batch_build_graph_observations_gpu
from .render import BottleneckRenderer
from .path_planner import update_agent_waypoints, get_waypoint_direction, get_waypoint_distance


class BottleneckInforMARLEnv(gym.Env):
    """2D 병목 환경 - InforMARL 기반"""
    
    def __init__(self, 
                 num_agents: int = 6,
                 agent_radius: float = 0.5,
                 corridor_width: float = 20.0,
                 corridor_height: float = 10.0,
                 bottleneck_width: float = 1.2,
                 bottleneck_position: float = 10.0,
                 sensing_radius: float = 3.0,
                 max_timesteps: int = 300,
                 config: dict = None,  # 🔥 YAML 설정을 받을 수 있게
                 gpu_id: int = None,   # 🚀 서버 GPU ID 지정
                 force_cpu: bool = False):  # CPU 강제 사용
        
        super().__init__()
        
        # 🚀 GPU 환경 설정
        setup_gpu_environment()
        self.device = get_device(gpu_id=gpu_id, force_cpu=force_cpu)
        
        # 🔥 YAML 설정이 있으면 우선 적용
        if config is not None:
            self.num_agents = config.get('num_agents', num_agents)
            self.agent_radius = config.get('agent_radius', agent_radius)
            self.corridor_width = config.get('corridor_width', corridor_width)
            self.corridor_height = config.get('corridor_height', corridor_height)
            self.bottleneck_width = config.get('bottleneck_width', bottleneck_width)
            self.bottleneck_position = config.get('bottleneck_position', bottleneck_position)
            self.sensing_radius = config.get('sensing_radius', sensing_radius)
            self.max_timesteps = config.get('max_timesteps', max_timesteps)
            
            # 성능 설정
            self.use_gpu_graph = config.get('use_gpu_graph', False)
            self.include_obstacles_in_gnn = config.get('include_obstacles_in_gnn', True)
            force_cpu = config.get('force_cpu', force_cpu)
        else:
            # 기본값 사용
            self.num_agents = num_agents
            self.agent_radius = agent_radius
            self.corridor_width = corridor_width
            self.corridor_height = corridor_height
            self.bottleneck_width = bottleneck_width
            self.bottleneck_position = bottleneck_position
            self.sensing_radius = sensing_radius
            self.max_timesteps = max_timesteps
            self.use_gpu_graph = False  # 기본값: CPU 그래프
            self.include_obstacles_in_gnn = True  # 기본값: 장애물 포함
        
        # 행동 공간: [위, 아래, 왼쪽, 오른쪽]
        self.action_space = spaces.Discrete(4)
        
        # 환경 상태
        self.agents: List[Agent2D] = []
        self.landmarks: List[Landmark2D] = []
        self.obstacles: List[Obstacle2D] = []
        self.informarl_agents: List[InforMARLAgent] = []
        
        # 공유 GNN과 옵티마이저
        self.shared_gnn = None
        self.gnn_optimizer = None
        
        # 시뮬레이션 상태
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        
        # 렌더링
        self.renderer = BottleneckRenderer()
        
    def reset(self) -> List[Data]:
        """환경 리셋"""
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        
        # 벽 여백 계산
        obstacle_spacing = (self.agent_radius * 2) / 3
        obstacle_radius = obstacle_spacing / 2
        wall_margin = obstacle_radius * 3
        
        # 환경 객체 생성
        self.agents, self.landmarks = create_agents_and_landmarks(
            self.num_agents, self.corridor_width, self.corridor_height,
            self.agent_radius, wall_margin
        )
        
        self.obstacles = create_obstacles(
            self.corridor_width, self.corridor_height,
            self.bottleneck_position, self.bottleneck_width,
            self.agent_radius
        )
        
        # InforMARL 에이전트 생성 (GPU 사용)
        self.informarl_agents = []
        for i in range(self.num_agents):
            informarl_agent = InforMARLAgent(agent_id=i, device=self.device)
            self.informarl_agents.append(informarl_agent)
        
        # 공유 GNN 초기화 (GPU 사용)
        if self.shared_gnn is None:
            self.shared_gnn = GraphNeuralNetwork().to(self.device)
            self.gnn_optimizer = torch.optim.Adam(self.shared_gnn.parameters(), lr=0.003)
        
        # 🚀 공유 Actor/Critic 초기화 (진짜 배치 처리용)
        from ..models.policy import Actor, Critic
        if not hasattr(self, 'shared_actor'):
            self.shared_actor = Actor(obs_dim=9, action_dim=4).to(self.device)  # waypoint 포함 obs_dim
            self.shared_critic = Critic().to(self.device)
            self.shared_policy_optimizer = torch.optim.Adam(
                list(self.shared_actor.parameters()) + list(self.shared_critic.parameters()), 
                lr=0.003
            )
        
        # 에이전트 waypoint 초기화
        update_agent_waypoints(
            self.agents, self.landmarks, self.corridor_width, self.corridor_height,
            self.bottleneck_position, self.bottleneck_width
        )
        
        # 그래프 생성 (설정에 따라 GPU/CPU 선택)
        if self.use_gpu_graph:
            try:
                return batch_build_graph_observations_gpu(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius, 
                    self.device, self.include_obstacles_in_gnn
                )
            except Exception as e:
                print(f"GPU graph building failed in reset, using CPU: {e}")
                return build_graph_observations(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius,
                    self.include_obstacles_in_gnn
                )
        else:
            return build_graph_observations(
                self.agents, self.landmarks, self.obstacles, self.sensing_radius,
                self.include_obstacles_in_gnn
            )
    
    def step(self, actions: List[int] = None):
        """환경 스텝 실행"""
        self.timestep += 1
        
        # 그래프 관측 생성 (설정에 따라 GPU/CPU 선택)
        if self.use_gpu_graph:
            try:
                graph_obs = batch_build_graph_observations_gpu(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius, 
                    self.device, self.include_obstacles_in_gnn
                )
            except Exception as e:
                print(f"GPU graph building failed, using CPU: {e}")
                graph_obs = build_graph_observations(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius,
                    self.include_obstacles_in_gnn
                )
        else:
            graph_obs = build_graph_observations(
                self.agents, self.landmarks, self.obstacles, self.sensing_radius,
                self.include_obstacles_in_gnn
            )
        
        # waypoint 업데이트 (5스텝마다로 줄임 - 성능 최적화)
        if self.timestep % 5 == 0:
            update_agent_waypoints(
                self.agents, self.landmarks, self.corridor_width, self.corridor_height,
                self.bottleneck_position, self.bottleneck_width
            )
        
        # 행동 선택 (배치 처리)
        if actions is None:
            actions, log_probs, values = self._get_batch_actions(graph_obs, training=True)
        else:
            log_probs = [0.0] * len(actions)
            values = [0.0] * len(actions)
        
        # 🚀 GPU 배치 물리 계산 (기존 CPU 방식보다 훨씬 빠름)
        try:
            # GPU에서 행동 실행 (배치)
            new_velocities = batch_execute_actions_gpu(self.agents, actions, self.device)
            
            # GPU에서 위치 업데이트 (배치)
            collision_count = batch_update_positions_gpu(
                self.agents, new_velocities, self.obstacles, 
                self.corridor_width, self.corridor_height,
                self.bottleneck_position, self.bottleneck_width, self.device
            )
        except Exception as e:
            # GPU 실패 시 CPU 백업
            print(f"GPU physics failed, using CPU: {e}")
            for i, action in enumerate(actions):
                execute_action(self.agents[i], action)
            
            collision_count = update_positions(
                self.agents, self.obstacles, self.corridor_width, self.corridor_height,
                self.bottleneck_position, self.bottleneck_width
            )
        self.collision_count += collision_count
        
        # 보상 계산 (GPU 병렬 처리로 성능 최적화)
        try:
            rewards = calculate_waypoint_rewards_gpu(self.agents, self.landmarks, self.device)
        except Exception as e:
            print(f"GPU reward calculation failed, using CPU: {e}")
            rewards = calculate_waypoint_rewards(self.agents, self.landmarks)
        
        # 성공 카운트 업데이트
        for agent in self.agents:
            target = self.landmarks[agent.target_id]
            if agent.get_distance_to(target.x, target.y) < target.radius:
                self.success_count += 1
        
        # 경험 저장
        if actions is not None:
            for i, (action, log_prob, value, reward) in enumerate(zip(actions, log_probs, values, rewards)):
                obs = self._get_local_observation(i)
                experience = {
                    'graph_data': graph_obs[i],
                    'local_obs': torch.tensor(obs, dtype=torch.float32),
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward
                }
                self.informarl_agents[i].store_experience(experience)
        
        # 🚀 성능 최적화: step 끝 그래프 생성 제거 (다음 스텝에서 어차피 새로 생성)
        done = self._is_done()
        info = self._get_info()
        
        # 빈 리스트 반환 (실제로는 사용되지 않음)
        return [], rewards, done, info
    
    def _get_batch_actions(self, graph_observations: List[Data], training: bool = True):
        """개별 그래프 처리 - 센싱 범위 제한 유지하면서 GPU 사용"""
        device = self.device
        
        actions = []
        log_probs = []
        values = []
        
        # 🚀 로컬 관측을 배치로 GPU 전송 (여전히 최적화 가능)
        all_local_obs = []
        for i in range(self.num_agents):
            obs = self._get_local_observation(i)
            all_local_obs.append(obs)
        local_obs_batch = torch.tensor(all_local_obs, dtype=torch.float32).to(device, non_blocking=True)
        
        # 🚀 배치 처리로 모든 그래프를 한 번에 GNN 통과
        batch_graphs = Batch.from_data_list(graph_observations).to(device, non_blocking=True)
        all_embeddings = self.shared_gnn(batch_graphs)
        
        # 각 그래프별로 임베딩 분리
        agent_embeddings = []
        global_embeddings = []
        
        current_idx = 0
        for i in range(self.num_agents):
            graph_size = len(graph_observations[i].x)
            
            # 이 그래프의 임베딩 추출
            graph_embeddings = all_embeddings[current_idx:current_idx + graph_size]
            
            # 에이전트 자신의 임베딩 (보통 첫 번째 노드)
            ego_embedding = graph_embeddings[i] if i < len(graph_embeddings) else graph_embeddings[0]
            agent_embeddings.append(ego_embedding)
            
            # 전역 집계를 위한 모든 에이전트 노드들의 평균
            # 센싱 범위 내 에이전트들만 포함 (논문의 핵심!)
            agent_indices = []
            for j, entity_type in enumerate(graph_observations[i].entity_type):
                if entity_type == 0:  # agent 타입
                    agent_indices.append(j)
            
            if agent_indices:
                agent_nodes = graph_embeddings[agent_indices]
                global_agg = agent_nodes.mean(dim=0)
            else:
                global_agg = ego_embedding
            
            global_embeddings.append(global_agg)
            
            current_idx += graph_size
        
        # GPU에서 배치 처리
        agent_embeddings_batch = torch.stack(agent_embeddings)
        global_embeddings_batch = torch.stack(global_embeddings)
        
        # 🚀 진짜 배치 처리 - 모든 에이전트를 한번에 처리
        if training:
            # 🚀 공유 Critic으로 모든 에이전트의 값 함수를 한번에 계산
            global_values = self.shared_critic(global_embeddings_batch)  # [N, 1]
            
            # 🚀 공유 Actor로 모든 에이전트의 행동 확률을 한번에 계산
            all_action_probs = self.shared_actor(local_obs_batch, agent_embeddings_batch)  # [N, 4]
            
            # 🚀 배치로 행동 선택
            action_dists = torch.distributions.Categorical(all_action_probs)
            sampled_actions = action_dists.sample()  # [N]
            log_probs_tensor = action_dists.log_prob(sampled_actions)  # [N]
            
            actions = sampled_actions.cpu().tolist()
            log_probs = log_probs_tensor.cpu().tolist()
            values = global_values.squeeze().cpu().tolist()
        else:
            # 🚀 평가 시도 배치 처리
            all_action_probs = self.shared_actor(local_obs_batch, agent_embeddings_batch)  # [N, 4]
            
            # 결정적 행동 선택 (가장 높은 확률)
            sampled_actions = torch.argmax(all_action_probs, dim=1)  # [N]
            actions = sampled_actions.cpu().tolist()
            log_probs = [0.0] * len(actions)
            values = [0.0] * len(actions)
        
        return actions, log_probs, values
    
    def _update_shared_networks(self):
        """🚀 공유 네트워크 배치 업데이트 - 모든 에이전트 경험을 한번에 처리"""
        # 모든 에이전트의 경험을 모음
        all_experiences = []
        for agent in self.informarl_agents:
            if len(agent.memory) >= 16:  # 최소 경험이 있는 에이전트만
                experiences = list(agent.memory)
                all_experiences.extend(experiences[-32:])  # 최근 32개만 사용
        
        if len(all_experiences) < 32:  # 전체 경험이 너무 적으면 스킵
            return
        
        # 🚀 배치 샘플링
        import random
        batch_size = min(64, len(all_experiences))  # 더 큰 배치 사이즈
        batch = random.sample(all_experiences, batch_size)
        
        # 배치 데이터 준비
        device = self.device
        graph_data_list = [exp['graph_data'] for exp in batch]
        local_obs = torch.stack([exp['local_obs'] for exp in batch]).to(device)
        actions = torch.tensor([exp['action'] for exp in batch]).to(device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch], dtype=torch.float32).to(device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(device)
        values = torch.tensor([exp['value'] for exp in batch], dtype=torch.float32).to(device)
        
        # GAE 계산
        advantages = self._compute_gae(rewards, values).to(device)
        returns = (advantages + values).to(device)
        
        # 🚀 PPO 업데이트 (더 적은 에폭으로 속도 향상)
        for _ in range(2):  # 4→2로 줄임
            # 공유 GNN으로 그래프 데이터 처리
            from torch_geometric.data import Batch
            batch_graphs = Batch.from_data_list(graph_data_list).to(device)
            node_embeddings = self.shared_gnn(batch_graphs)
            
            # 에이전트 임베딩 추출 (simplified)
            nodes_per_graph = len(graph_data_list[0].x) if graph_data_list else 1
            agent_embeddings = []
            global_embeddings = []
            
            for i in range(len(batch)):
                start_idx = i * nodes_per_graph
                end_idx = min(start_idx + nodes_per_graph, len(node_embeddings))
                
                if start_idx < len(node_embeddings):
                    agent_emb = node_embeddings[start_idx]
                    agent_embeddings.append(agent_emb)
                    
                    # 글로벌 집계 (단순화)
                    graph_nodes = node_embeddings[start_idx:end_idx]
                    global_agg = graph_nodes.mean(dim=0) if len(graph_nodes) > 0 else agent_emb
                    global_embeddings.append(global_agg)
                else:
                    # 폴백
                    agent_embeddings.append(node_embeddings[0] if len(node_embeddings) > 0 else torch.zeros(64, device=device))
                    global_embeddings.append(node_embeddings[0] if len(node_embeddings) > 0 else torch.zeros(64, device=device))
            
            agent_embeddings = torch.stack(agent_embeddings)
            global_embeddings = torch.stack(global_embeddings)
            
            # 🚀 공유 네트워크로 배치 처리
            action_probs = self.shared_actor(local_obs, agent_embeddings)
            current_values = self.shared_critic(global_embeddings).squeeze()
            
            # PPO 손실 계산
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss (clipped)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(current_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # 🚀 공유 네트워크 업데이트
            self.shared_policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.shared_actor.parameters()) + list(self.shared_critic.parameters()), 
                0.5
            )
            self.shared_policy_optimizer.step()
    
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
    
    def _get_local_observation(self, agent_id: int) -> List[float]:
        """에이전트의 로컬 관측 (waypoint 포함)"""
        agent = self.agents[agent_id]
        target = self.landmarks[agent.target_id]
        
        # 기본 관측
        obs = [
            agent.x / self.sensing_radius,    # sensing_radius로 정규화된 위치
            agent.y / self.sensing_radius,
            agent.vx / agent.max_speed,       # 정규화된 속도
            agent.vy / agent.max_speed,
            (target.x - agent.x) / self.sensing_radius,  # 상대 목표 위치
            (target.y - agent.y) / self.sensing_radius
        ]
        
        # waypoint 정보 추가
        waypoint_direction = get_waypoint_direction(agent)
        waypoint_distance = get_waypoint_distance(agent)
        
        obs.extend([
            waypoint_direction[0],  # waypoint 방향 x
            waypoint_direction[1],  # waypoint 방향 y  
            waypoint_distance / self.sensing_radius  # 정규화된 waypoint 거리
        ])
        
        return obs
    
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
    
    def render(self, mode='human', show_waypoints=True):
        """환경 렌더링"""
        self.renderer.render(
            self.agents, self.landmarks, self.obstacles,
            self.corridor_width, self.corridor_height,
            self.bottleneck_position, self.bottleneck_width,
            self.timestep, self.success_count, self.collision_count,
            show_waypoints=show_waypoints
        )