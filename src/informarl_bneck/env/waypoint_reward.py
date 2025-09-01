"""
Waypoint-based reward calculation for bottleneck navigation
"""
import numpy as np
import math
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
        
        # 4. 충돌 페널티 (크게!)
        collision_penalty = getattr(agent, 'collision_penalty_timer', 0)
        if collision_penalty > 0:
            reward -= 50.0  # 큰 페널티로 양보 학습 유도
        
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