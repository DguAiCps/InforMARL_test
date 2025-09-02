"""
Waypoint-based reward calculation for bottleneck navigation
"""
import numpy as np
import math
import torch
from typing import List
from ..utils.types import Agent2D, Landmark2D
from .path_planner import get_waypoint_distance, get_waypoint_direction


def calculate_waypoint_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """Waypoint 기반 보상 계산 - 훨씬 학습하기 쉬운 구조"""
    rewards = []
    
    for agent in agents:
        reward = 0.0
        target = landmarks[agent.target_id]
        final_distance = agent.get_distance_to(target.x, target.y)
        
        # 1. 최종 목표 도달 - 매우 큰 보상
        if final_distance < target.radius:
            reward += 200.0
            rewards.append(reward)
            continue
        
        # 2. 기본 생존 보상 (매 스텝마다 양수)
        reward += 0.5
        
        # 3. Waypoint 진행 보상 (핵심!)
        if hasattr(agent, 'current_waypoint'):
            waypoint_distance = get_waypoint_distance(agent)
            waypoint_direction = get_waypoint_direction(agent)
            
            # 3-1. Waypoint에 가까워질수록 보상
            max_distance = 20.0  # 맵 크기 기준
            proximity_reward = max(0, (max_distance - waypoint_distance) / max_distance) * 2.0
            reward += proximity_reward
            
            # 3-2. Waypoint 방향으로 움직이면 추가 보상
            agent_velocity = np.array([agent.vx, agent.vy])
            velocity_magnitude = np.linalg.norm(agent_velocity)
            
            if velocity_magnitude > 0.05:  # 움직이고 있을 때
                agent_direction = agent_velocity / velocity_magnitude
                waypoint_dir_vec = np.array(waypoint_direction)
                
                # 방향 일치도 (-1 ~ 1)
                direction_alignment = np.dot(agent_direction, waypoint_dir_vec)
                if direction_alignment > 0:  # 올바른 방향으로 움직일 때만
                    reward += direction_alignment * velocity_magnitude * 3.0
            
            # 3-3. Waypoint 도달 보너스
            if waypoint_distance < 1.0:  # waypoint에 충분히 가까우면
                reward += 10.0
        
        # 4. 충돌 페널티 (양보 학습을 위해 적절한 강도)
        collision_penalty = getattr(agent, 'collision_penalty_timer', 0)
        if collision_penalty > 0:
            reward -= 35.0  # 양보 행동 학습을 위한 적절한 페널티
        
        # 5. 정지 페널티 (작게)
        agent_velocity = np.array([agent.vx, agent.vy])
        if np.linalg.norm(agent_velocity) < 0.05:
            reward -= 1.0  # 작은 페널티
        
        # 6. 최종 목표 방향 보너스 (waypoint와 별도)
        target_direction = np.array([target.x - agent.x, target.y - agent.y])
        target_distance_norm = np.linalg.norm(target_direction)
        
        if target_distance_norm > 0.1:
            target_direction = target_direction / target_distance_norm
            agent_velocity = np.array([agent.vx, agent.vy])
            velocity_magnitude = np.linalg.norm(agent_velocity)
            
            if velocity_magnitude > 0.05:
                agent_direction = agent_velocity / velocity_magnitude
                final_alignment = np.dot(agent_direction, target_direction)
                if final_alignment > 0:
                    reward += final_alignment * 0.5  # 작은 추가 보너스
        
        rewards.append(reward)
    
    return rewards


def calculate_waypoint_rewards_gpu(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                 device: torch.device) -> List[float]:
    """🚀 GPU 병렬 웨이포인트 보상 계산 - 대폭 성능 향상"""
    num_agents = len(agents)
    if num_agents == 0:
        return []
    
    # 모든 에이전트 데이터를 텐서로 배치 변환
    agent_positions = torch.tensor([[agent.x, agent.y] for agent in agents], 
                                  dtype=torch.float32, device=device)  # [N, 2]
    agent_velocities = torch.tensor([[agent.vx, agent.vy] for agent in agents], 
                                   dtype=torch.float32, device=device)  # [N, 2]
    
    # 타겟 정보 배치 변환
    target_positions = torch.tensor([[landmarks[agent.target_id].x, landmarks[agent.target_id].y] 
                                   for agent in agents], dtype=torch.float32, device=device)  # [N, 2]
    target_radii = torch.tensor([landmarks[agent.target_id].radius for agent in agents], 
                               dtype=torch.float32, device=device)  # [N]
    
    # 웨이포인트 정보 배치 변환
    waypoint_positions = torch.tensor([getattr(agent, 'current_waypoint', (agent.x, agent.y)) 
                                     for agent in agents], dtype=torch.float32, device=device)  # [N, 2]
    
    # 충돌 페널티 정보
    collision_penalties = torch.tensor([getattr(agent, 'collision_penalty_timer', 0) 
                                      for agent in agents], dtype=torch.float32, device=device)  # [N]
    
    # 🚀 GPU 배치 계산 시작
    rewards = torch.zeros(num_agents, dtype=torch.float32, device=device)
    
    # 1. 최종 목표까지 거리 계산 (배치)
    final_distances = torch.norm(agent_positions - target_positions, dim=1)  # [N]
    goal_reached = final_distances < target_radii  # [N] boolean
    
    # 목표 도달 시 200점 부여하고 early return
    rewards[goal_reached] = 200.0
    active_mask = ~goal_reached  # 목표에 도달하지 않은 에이전트들
    
    if active_mask.sum() == 0:  # 모든 에이전트가 목표 도달
        return rewards.cpu().tolist()
    
    # 2. 기본 생존 보상 (+0.5)
    rewards[active_mask] += 0.5
    
    # 3. 웨이포인트 진행 보상들 (active 에이전트들만)
    active_agent_pos = agent_positions[active_mask]  # [M, 2]
    active_waypoint_pos = waypoint_positions[active_mask]  # [M, 2]
    active_velocities = agent_velocities[active_mask]  # [M, 2]
    active_target_pos = target_positions[active_mask]  # [M, 2]
    
    # 3-1. 웨이포인트 근접 보상
    waypoint_distances = torch.norm(active_agent_pos - active_waypoint_pos, dim=1)  # [M]
    max_distance = 20.0
    proximity_rewards = torch.clamp((max_distance - waypoint_distances) / max_distance, min=0) * 2.0
    rewards[active_mask] += proximity_rewards
    
    # 3-2. 웨이포인트 방향 이동 보상
    velocity_magnitudes = torch.norm(active_velocities, dim=1)  # [M]
    moving_mask = velocity_magnitudes > 0.05  # 움직이고 있는 에이전트들
    
    if moving_mask.sum() > 0:
        # 방향 벡터들
        agent_directions = active_velocities[moving_mask] / velocity_magnitudes[moving_mask].unsqueeze(1)  # [K, 2]
        waypoint_directions = active_waypoint_pos[moving_mask] - active_agent_pos[moving_mask]  # [K, 2]
        waypoint_dir_norms = torch.norm(waypoint_directions, dim=1)  # [K]
        
        valid_waypoint_mask = waypoint_dir_norms > 0.01
        if valid_waypoint_mask.sum() > 0:
            waypoint_directions[valid_waypoint_mask] = (waypoint_directions[valid_waypoint_mask] / 
                                                       waypoint_dir_norms[valid_waypoint_mask].unsqueeze(1))
            
            # 방향 일치도 계산 (내적)
            direction_alignments = torch.sum(agent_directions * waypoint_directions, dim=1)  # [K]
            positive_alignment = torch.clamp(direction_alignments, min=0)  # 양수만
            
            # 보상 계산
            direction_rewards = positive_alignment * velocity_magnitudes[moving_mask] * 3.0
            
            # active_mask 내에서 moving_mask인 인덱스들에 보상 추가
            active_indices = torch.where(active_mask)[0]
            moving_indices = active_indices[moving_mask]
            rewards[moving_indices] += direction_rewards
    
    # 3-3. 웨이포인트 도달 보너스
    waypoint_close = waypoint_distances < 1.0
    rewards[torch.where(active_mask)[0][waypoint_close]] += 10.0
    
    # 4. 충돌 페널티 (-35점)
    collision_mask = collision_penalties > 0
    rewards[collision_mask] -= 35.0
    
    # 5. 정지 페널티 (-1점)
    all_velocity_magnitudes = torch.norm(agent_velocities, dim=1)
    stationary_mask = all_velocity_magnitudes < 0.05
    rewards[stationary_mask] -= 1.0
    
    # 6. 최종 목표 방향 보너스
    target_directions = target_positions - agent_positions  # [N, 2]
    target_distances = torch.norm(target_directions, dim=1)  # [N]
    
    valid_target_mask = target_distances > 0.1
    if valid_target_mask.sum() > 0:
        target_directions[valid_target_mask] = (target_directions[valid_target_mask] / 
                                               target_distances[valid_target_mask].unsqueeze(1))
        
        moving_all_mask = all_velocity_magnitudes > 0.05
        final_bonus_mask = valid_target_mask & moving_all_mask
        
        if final_bonus_mask.sum() > 0:
            agent_dirs_all = agent_velocities[final_bonus_mask] / all_velocity_magnitudes[final_bonus_mask].unsqueeze(1)
            final_alignments = torch.sum(agent_dirs_all * target_directions[final_bonus_mask], dim=1)
            positive_final_alignments = torch.clamp(final_alignments, min=0)
            rewards[final_bonus_mask] += positive_final_alignments * 0.5
    
    return rewards.cpu().tolist()