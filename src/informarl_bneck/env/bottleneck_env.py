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
from .graph_builder import build_graph_observations, batch_build_graph_observations_gpu
from .render import BottleneckRenderer


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
        
        # 그래프 생성 (설정에 따라 GPU/CPU 선택)
        if self.use_gpu_graph:
            try:
                return batch_build_graph_observations_gpu(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius, self.device
                )
            except Exception as e:
                print(f"GPU graph building failed in reset, using CPU: {e}")
                return build_graph_observations(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius
                )
        else:
            return build_graph_observations(
                self.agents, self.landmarks, self.obstacles, self.sensing_radius
            )
    
    def step(self, actions: List[int] = None):
        """환경 스텝 실행"""
        self.timestep += 1
        if self.timestep % 10 == 0:  # 10 스텝마다 진행상황 출력
            print(f"  Step {self.timestep}...")
        
        # 그래프 관측 생성 (설정에 따라 GPU/CPU 선택)
        if self.use_gpu_graph:
            try:
                graph_obs = batch_build_graph_observations_gpu(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius, self.device
                )
            except Exception as e:
                print(f"GPU graph building failed, using CPU: {e}")
                graph_obs = build_graph_observations(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius
                )
        else:
            graph_obs = build_graph_observations(
                self.agents, self.landmarks, self.obstacles, self.sensing_radius
            )
        
        # 행동 선택 (배치 처리)
        if actions is None:
            if self.timestep % 50 == 0:  # 50 스텝마다 행동 선택 과정 출력
                print(f"    Getting actions at step {self.timestep}...")
            actions, log_probs, values = self._get_batch_actions(graph_obs, training=True)
        else:
            log_probs = [0.0] * len(actions)
            values = [0.0] * len(actions)
        
        # 🚀 GPU 배치 물리 계산 (기존 CPU 방식보다 훨씬 빠름)
        try:
            # GPU에서 행동 실행 (배치)
            new_velocities, new_penalties = batch_execute_actions_gpu(self.agents, actions, self.device)
            
            # 페널티 타이머 업데이트
            for i, agent in enumerate(self.agents):
                agent.collision_penalty_timer = int(new_penalties[i].item())
            
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
        
        # 보상 계산
        rewards = calculate_rewards(self.agents, self.landmarks)
        
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
        
        # step 끝에서 새로운 관측 생성 (설정에 따라 GPU/CPU 선택)
        if self.use_gpu_graph:
            try:
                new_obs = batch_build_graph_observations_gpu(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius, self.device
                )
            except Exception as e:
                print(f"GPU graph building failed in step end, using CPU: {e}")
                new_obs = build_graph_observations(
                    self.agents, self.landmarks, self.obstacles, self.sensing_radius
                )
        else:
            new_obs = build_graph_observations(
                self.agents, self.landmarks, self.obstacles, self.sensing_radius
            )
        done = self._is_done()
        info = self._get_info()
        
        return new_obs, rewards, done, info
    
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
        
        # 각 에이전트별로 개별 그래프 처리
        agent_embeddings = []
        global_embeddings = []
        
        for i in range(self.num_agents):
            # 각 그래프를 개별적으로 GPU에서 처리
            graph_data = graph_observations[i].to(device, non_blocking=True)
            node_embeddings = self.shared_gnn(graph_data)
            
            # 에이전트 자신의 임베딩 (첫 번째 노드는 항상 ego agent)
            ego_embedding = node_embeddings[i] if i < len(node_embeddings) else node_embeddings[0]
            agent_embeddings.append(ego_embedding)
            
            # 전역 집계를 위한 모든 에이전트 노드들의 평균
            # 센싱 범위 내 에이전트들만 포함 (논문의 핵심!)
            agent_indices = []
            for j, entity_type in enumerate(graph_data.entity_type):
                if entity_type == 0:  # agent 타입
                    agent_indices.append(j)
            
            if agent_indices:
                agent_nodes = node_embeddings[agent_indices]
                global_agg = agent_nodes.mean(dim=0)
            else:
                global_agg = ego_embedding
            
            global_embeddings.append(global_agg)
        
        # GPU에서 배치 처리
        agent_embeddings_batch = torch.stack(agent_embeddings)
        global_embeddings_batch = torch.stack(global_embeddings)
        
        if training:
            # Critic으로 값 함수 계산
            global_values = self.informarl_agents[0].critic(global_embeddings_batch)
            
            for i, agent in enumerate(self.informarl_agents):
                # Actor: 로컬 관측 + 집계 정보
                local_obs = local_obs_batch[i].unsqueeze(0)
                agg_info = agent_embeddings_batch[i].unsqueeze(0)
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
            for i, agent in enumerate(self.informarl_agents):
                local_obs = local_obs_batch[i].unsqueeze(0)
                agg_info = agent_embeddings_batch[i].unsqueeze(0)
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
    
    def render(self, mode='human'):
        """환경 렌더링"""
        self.renderer.render(
            self.agents, self.landmarks, self.obstacles,
            self.corridor_width, self.corridor_height,
            self.bottleneck_position, self.bottleneck_width,
            self.timestep, self.success_count, self.collision_count
        )