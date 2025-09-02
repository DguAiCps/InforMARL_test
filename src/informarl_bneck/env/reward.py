"""
Reward calculation for bottleneck navigation
"""
import numpy as np
import math
from typing import List
from ..utils.types import Agent2D, Landmark2D


def calculate_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """개선된 보상 계산 - 좌우 반복 이동 방지"""
    rewards = []
    
    for agent in agents:
        reward = 0.0
        target = landmarks[agent.target_id]
        distance = agent.get_distance_to(target.x, target.y)
        
        # 목표 도달 - 큰 보상
        if distance < target.radius:
            reward += 100.0
            rewards.append(reward)
            continue
        
        # 1. 거리 기반 기본 보상 (음수로 시작해서 가까워질수록 덜 나쁨)
        reward -= distance * 0.5
        
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
        
        # 5. 충돌 페널티 - 충돌 시 마이너스 보상
        collision_penalty = getattr(agent, 'collision_penalty_timer', 0)
        if collision_penalty > 0:
            reward -= 2.0  # 충돌 시 적당한 페널티
        
        # 6. 시간 페널티 (너무 오래 걸리면)
        reward -= 0.01  # 매 스텝마다 작은 시간 페널티
        
        rewards.append(reward)
    
    return rewards