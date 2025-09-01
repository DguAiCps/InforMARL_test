"""
Simple waypoint-based path planning for InforMARL bottleneck navigation
"""
import math
from typing import List, Tuple
from ..utils.types import Agent2D, Landmark2D


def generate_basic_waypoints(agent: Agent2D, target: Landmark2D, 
                           corridor_width: float, corridor_height: float,
                           bottleneck_position: float, bottleneck_width: float) -> List[Tuple[float, float]]:
    """에이전트를 위한 기본 waypoint 경로 생성"""
    
    start_x, start_y = agent.x, agent.y
    goal_x, goal_y = target.x, target.y
    
    waypoints = [(start_x, start_y)]  # 시작점
    
    # 병목 통과가 필요한지 확인
    need_bottleneck = False
    
    # 에이전트와 목표가 병목 양쪽에 있는지 체크
    agent_side = "left" if start_x < bottleneck_position else "right"
    goal_side = "left" if goal_x < bottleneck_position else "right"
    
    if agent_side != goal_side:
        need_bottleneck = True
    
    if need_bottleneck:
        # 병목 통과 경로
        bottleneck_center_y = corridor_height / 2
        
        if agent_side == "left":
            # 왼쪽 → 오른쪽으로 이동
            # 1. 병목 입구로 이동
            waypoints.append((bottleneck_position - 1.0, bottleneck_center_y))
            # 2. 병목 통과
            waypoints.append((bottleneck_position + 1.0, bottleneck_center_y))
            # 3. 목표로 이동
            waypoints.append((goal_x, goal_y))
        else:
            # 오른쪽 → 왼쪽으로 이동
            # 1. 병목 입구로 이동
            waypoints.append((bottleneck_position + 1.0, bottleneck_center_y))
            # 2. 병목 통과
            waypoints.append((bottleneck_position - 1.0, bottleneck_center_y))
            # 3. 목표로 이동
            waypoints.append((goal_x, goal_y))
    else:
        # 같은 쪽에 있으면 직선 경로
        waypoints.append((goal_x, goal_y))
    
    return waypoints


def get_current_waypoint(agent: Agent2D, waypoints: List[Tuple[float, float]], 
                        waypoint_threshold: float = 1.0) -> Tuple[float, float]:
    """현재 에이전트가 향해야 할 waypoint 반환"""
    
    if not waypoints:
        return (agent.x, agent.y)  # 기본값
    
    agent_pos = (agent.x, agent.y)
    
    # 각 waypoint까지의 거리 확인
    for waypoint in waypoints:
        distance = math.sqrt((agent_pos[0] - waypoint[0])**2 + (agent_pos[1] - waypoint[1])**2)
        
        # 아직 도달하지 않은 첫 번째 waypoint 반환
        if distance > waypoint_threshold:
            return waypoint
    
    # 모든 waypoint에 도달했으면 마지막 waypoint (목표) 반환
    return waypoints[-1]


def update_agent_waypoints(agents: List[Agent2D], landmarks: List[Landmark2D],
                          corridor_width: float, corridor_height: float,
                          bottleneck_position: float, bottleneck_width: float):
    """모든 에이전트의 waypoint 업데이트"""
    
    for agent in agents:
        target = landmarks[agent.target_id]
        
        # waypoint가 없거나 목표가 변경되었으면 새로 생성
        if not hasattr(agent, 'waypoints') or agent.waypoints is None:
            agent.waypoints = generate_basic_waypoints(
                agent, target, corridor_width, corridor_height,
                bottleneck_position, bottleneck_width
            )
        
        # 현재 목표 waypoint 업데이트
        agent.current_waypoint = get_current_waypoint(agent, agent.waypoints)


def get_waypoint_direction(agent: Agent2D) -> Tuple[float, float]:
    """에이전트가 현재 waypoint로 향하는 방향 벡터 반환 (정규화됨)"""
    
    if not hasattr(agent, 'current_waypoint'):
        return (0.0, 0.0)
    
    waypoint_x, waypoint_y = agent.current_waypoint
    
    # 방향 벡터 계산
    dx = waypoint_x - agent.x
    dy = waypoint_y - agent.y
    
    # 정규화
    distance = math.sqrt(dx*dx + dy*dy)
    if distance > 0.01:
        return (dx / distance, dy / distance)
    else:
        return (0.0, 0.0)


def get_waypoint_distance(agent: Agent2D) -> float:
    """에이전트와 현재 waypoint 사이의 거리"""
    
    if not hasattr(agent, 'current_waypoint'):
        return 0.0
    
    waypoint_x, waypoint_y = agent.current_waypoint
    return math.sqrt((agent.x - waypoint_x)**2 + (agent.y - waypoint_y)**2)