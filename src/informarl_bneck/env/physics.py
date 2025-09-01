"""
Physics simulation for agent movement and collision handling
"""
import math
import torch
from typing import List, Dict, Any
from ..utils.types import Agent2D, Obstacle2D


def execute_action(agent: Agent2D, action: int):
    """행동 실행 - 충돌 페널티 적용"""
    
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


def update_positions(agents: List[Agent2D], obstacles: List[Obstacle2D], 
                    corridor_width: float, corridor_height: float,
                    bottleneck_position: float, bottleneck_width: float) -> int:
    """위치 업데이트 - 강화된 충돌 처리"""
    dt = 0.1
    collision_count = 0
    
    for agent in agents:
        new_x = agent.x + agent.vx * dt
        new_y = agent.y + agent.vy * dt
        
        # 통합된 충돌 체크
        collision_info = check_collision_detailed(
            agent, new_x, new_y, agents, obstacles,
            corridor_width, corridor_height, bottleneck_position, bottleneck_width
        )
        
        if not collision_info['has_collision']:
            agent.x = new_x
            agent.y = new_y
        else:
            # 충돌 시 강화된 처리
            handle_collision(agent, collision_info, corridor_width, corridor_height)
            collision_count += 1
    
    return collision_count


def check_collision_detailed(agent: Agent2D, new_x: float, new_y: float,
                           agents: List[Agent2D], obstacles: List[Obstacle2D],
                           corridor_width: float, corridor_height: float,
                           bottleneck_position: float, bottleneck_width: float) -> Dict[str, Any]:
    """상세 충돌 정보 체크"""
    agent_radius = agent.radius
    collision_info = {
        'has_collision': False,
        'collision_type': None,
        'collision_entity': None,
        'penetration_depth': 0.0
    }
    
    # 1. 환경 경계 체크
    if (new_x - agent_radius < 0 or new_x + agent_radius > corridor_width or
        new_y - agent_radius < 0 or new_y + agent_radius > corridor_height):
        collision_info.update({
            'has_collision': True,
            'collision_type': 'boundary',
            'penetration_depth': max(
                max(0, (agent_radius - new_x)),  # 왼쪽 경계
                max(0, (new_x + agent_radius - corridor_width)),  # 오른쪽 경계
                max(0, (agent_radius - new_y)),  # 아래 경계
                max(0, (new_y + agent_radius - corridor_height))  # 위 경계
            )
        })
        return collision_info
    
    # 2. 병목 벽 충돌 체크
    if not can_pass_through_bottleneck(new_x, new_y, agent_radius, 
                                     corridor_height, bottleneck_position, bottleneck_width):
        collision_info.update({
            'has_collision': True,
            'collision_type': 'bottleneck_wall'
        })
        return collision_info
    
    # 3. 장애물 충돌 체크
    for obstacle in obstacles:
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
    for other in agents:
        if other.id != agent.id:
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


def can_pass_through_bottleneck(x: float, y: float, radius: float,
                               corridor_height: float, bottleneck_position: float, 
                               bottleneck_width: float) -> bool:
    """병목 구간 통과 가능 여부 체크"""
    center_y = corridor_height / 2
    bottleneck_x = bottleneck_position
    
    # 병목 구역이 아니면 통과 가능
    if abs(x - bottleneck_x) > 1.0:
        return True
    
    # 병목 구역 내에서는 통로 폭 체크 (에이전트 반지름 고려)
    passage_top = center_y + bottleneck_width / 2
    passage_bottom = center_y - bottleneck_width / 2
    
    # 에이전트가 통로 안에 완전히 들어갈 수 있는지 체크
    agent_top = y + radius
    agent_bottom = y - radius
    
    return (agent_bottom >= passage_bottom and agent_top <= passage_top)


def handle_collision(agent: Agent2D, collision_info: Dict[str, Any], 
                    corridor_width: float, corridor_height: float):
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
        elif agent.x > corridor_width - margin:
            agent.x = corridor_width - margin
        if agent.y < margin:
            agent.y = margin
        elif agent.y > corridor_height - margin:
            agent.y = corridor_height - margin
            
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


# =============================================================================
# 🚀 GPU 배치 물리계산 함수들
# =============================================================================

def batch_execute_actions_gpu(agents: List[Agent2D], actions: List[int], device: torch.device) -> torch.Tensor:
    """GPU에서 모든 에이전트 행동을 배치로 처리"""
    num_agents = len(agents)
    
    # 현재 에이전트 상태를 텐서로 변환
    positions = torch.tensor([[agent.x, agent.y] for agent in agents], dtype=torch.float32).to(device)
    velocities = torch.tensor([[agent.vx, agent.vy] for agent in agents], dtype=torch.float32).to(device)
    penalties = torch.tensor([getattr(agent, 'collision_penalty_timer', 0) for agent in agents], dtype=torch.float32).to(device)
    max_speeds = torch.tensor([agent.max_speed for agent in agents], dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    
    # 충돌 페널티에 따른 속도 조정
    normal_speed = max_speeds * 0.5
    penalty_speed = max_speeds * 0.2
    current_speeds = torch.where(penalties > 0, penalty_speed, normal_speed)
    
    # 행동을 속도 변화로 변환 (배치)
    action_to_velocity_change = torch.tensor([
        [0, 1],    # 위
        [0, -1],   # 아래
        [-1, 0],   # 왼쪽
        [1, 0]     # 오른쪽
    ], dtype=torch.float32).to(device)
    
    velocity_changes = action_to_velocity_change[actions_tensor] * current_speeds.unsqueeze(1)
    new_velocities = velocities + velocity_changes
    
    # 최대 속도 제한 (배치)
    max_speed_limit = max_speeds.unsqueeze(1)
    new_velocities = torch.clamp(new_velocities, -max_speed_limit, max_speed_limit)
    
    # 속도 감쇠 (배치)
    decay_normal = torch.full_like(penalties, 0.9)
    decay_penalty = torch.full_like(penalties, 0.7)
    decay_factors = torch.where(penalties > 0, decay_penalty, decay_normal).unsqueeze(1)
    new_velocities = new_velocities * decay_factors
    
    # 페널티 타이머 감소
    new_penalties = torch.clamp(penalties - 1, min=0)
    
    return new_velocities, new_penalties


def batch_update_positions_gpu(agents: List[Agent2D], new_velocities: torch.Tensor, 
                              obstacles: List[Obstacle2D], corridor_width: float, 
                              corridor_height: float, bottleneck_position: float, 
                              bottleneck_width: float, device: torch.device, dt: float = 0.1) -> int:
    """GPU에서 모든 에이전트 위치를 배치로 업데이트"""
    num_agents = len(agents)
    
    # 현재 위치를 텐서로 변환
    positions = torch.tensor([[agent.x, agent.y] for agent in agents], dtype=torch.float32).to(device)
    radii = torch.tensor([agent.radius for agent in agents], dtype=torch.float32).to(device)
    
    # 새로운 위치 계산
    new_positions = positions + new_velocities * dt
    
    # 🚀 GPU에서 배치 충돌 검사
    collision_mask, collision_count = batch_check_collisions_gpu(
        new_positions, radii, obstacles, corridor_width, corridor_height, 
        bottleneck_position, bottleneck_width, device
    )
    
    # 충돌이 없는 에이전트만 위치 업데이트
    valid_positions = torch.where(collision_mask.unsqueeze(1), positions, new_positions)
    valid_velocities = torch.where(collision_mask.unsqueeze(1), torch.zeros_like(new_velocities), new_velocities)
    
    # 결과를 다시 에이전트 객체에 적용
    valid_positions_cpu = valid_positions.cpu().numpy()
    valid_velocities_cpu = valid_velocities.cpu().numpy()
    
    for i, agent in enumerate(agents):
        agent.x = float(valid_positions_cpu[i, 0])
        agent.y = float(valid_positions_cpu[i, 1])
        agent.vx = float(valid_velocities_cpu[i, 0])
        agent.vy = float(valid_velocities_cpu[i, 1])
        
        # 충돌한 에이전트에 페널티 추가
        if collision_mask[i].item():
            agent.collision_penalty_timer = getattr(agent, 'collision_penalty_timer', 0) + 3
    
    return int(collision_count.item())


def batch_check_collisions_gpu(positions: torch.Tensor, radii: torch.Tensor,
                              obstacles: List[Obstacle2D], corridor_width: float,
                              corridor_height: float, bottleneck_position: float,
                              bottleneck_width: float, device: torch.device) -> tuple:
    """GPU에서 배치 충돌 검사"""
    num_agents = positions.shape[0]
    collision_mask = torch.zeros(num_agents, dtype=torch.bool).to(device)
    
    # 1. 경계 충돌 검사 (배치)
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    boundary_collision = (
        (x_pos - radii < 0) | 
        (x_pos + radii > corridor_width) |
        (y_pos - radii < 0) | 
        (y_pos + radii > corridor_height)
    )
    collision_mask = collision_mask | boundary_collision
    
    # 2. 병목 벽 충돌 검사 (배치)
    bottleneck_collision = batch_check_bottleneck_collision_gpu(
        positions, radii, corridor_height, bottleneck_position, bottleneck_width, device
    )
    collision_mask = collision_mask | bottleneck_collision
    
    # 3. 에이전트 간 충돌 검사 (배치)
    if num_agents > 1:
        agent_collision = batch_check_agent_collisions_gpu(positions, radii, device)
        collision_mask = collision_mask | agent_collision
    
    # 4. 장애물 충돌 검사 (배치)
    if obstacles:
        obstacle_collision = batch_check_obstacle_collisions_gpu(positions, radii, obstacles, device)
        collision_mask = collision_mask | obstacle_collision
    
    collision_count = collision_mask.sum()
    return collision_mask, collision_count


def batch_check_bottleneck_collision_gpu(positions: torch.Tensor, radii: torch.Tensor,
                                        corridor_height: float, bottleneck_position: float,
                                        bottleneck_width: float, device: torch.device) -> torch.Tensor:
    """GPU에서 병목 충돌 검사"""
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    
    center_y = corridor_height / 2
    
    # 병목 구역에 있는 에이전트만 체크
    in_bottleneck_area = torch.abs(x_pos - bottleneck_position) <= 1.0
    
    # 통로 범위
    passage_top = center_y + bottleneck_width / 2
    passage_bottom = center_y - bottleneck_width / 2
    
    # 에이전트 범위 (반지름 고려)
    agent_top = y_pos + radii
    agent_bottom = y_pos - radii
    
    # 통로를 벗어난 에이전트
    outside_passage = (agent_bottom < passage_bottom) | (agent_top > passage_top)
    
    return in_bottleneck_area & outside_passage


def batch_check_agent_collisions_gpu(positions: torch.Tensor, radii: torch.Tensor, device: torch.device) -> torch.Tensor:
    """GPU에서 에이전트 간 충돌 검사"""
    num_agents = positions.shape[0]
    
    # 모든 에이전트 쌍 간의 거리 계산
    distance_matrix = torch.cdist(positions, positions, p=2)
    
    # 각 에이전트 쌍의 최소 거리 (반지름의 합)
    radii_sum_matrix = radii.unsqueeze(0) + radii.unsqueeze(1)
    
    # 대각선 제거 (자기 자신과의 거리)
    mask = torch.eye(num_agents, dtype=torch.bool, device=device)
    distance_matrix = distance_matrix.masked_fill(mask, float('inf'))
    
    # 충돌 검사
    collision_matrix = distance_matrix < radii_sum_matrix
    collision_mask = collision_matrix.any(dim=1)
    
    return collision_mask


def batch_check_obstacle_collisions_gpu(positions: torch.Tensor, radii: torch.Tensor,
                                       obstacles: List[Obstacle2D], device: torch.device) -> torch.Tensor:
    """GPU에서 장애물 충돌 검사"""
    num_agents = positions.shape[0]
    collision_mask = torch.zeros(num_agents, dtype=torch.bool).to(device)
    
    # 각 장애물에 대해 검사
    for obstacle in obstacles:
        obstacle_pos = torch.tensor([obstacle.x, obstacle.y], dtype=torch.float32).to(device)
        obstacle_radius = obstacle.radius
        
        # 모든 에이전트와 이 장애물 간의 거리
        distances = torch.norm(positions - obstacle_pos, dim=1)
        min_distances = radii + obstacle_radius
        
        # 충돌 검사
        obstacle_collision = distances < min_distances
        collision_mask = collision_mask | obstacle_collision
    
    return collision_mask