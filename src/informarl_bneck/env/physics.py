"""
Physics simulation for agent movement and collision handling
"""
import math
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