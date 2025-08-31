"""
Map generation for bottleneck environment
"""
import numpy as np
from typing import List
from ..utils.types import Agent2D, Landmark2D, Obstacle2D


def create_agents_and_landmarks(num_agents: int, corridor_width: float, corridor_height: float,
                               agent_radius: float, wall_margin: float) -> tuple[List[Agent2D], List[Landmark2D]]:
    """에이전트와 목표 지점 생성"""
    agents = []
    landmarks = []
    
    # 목표 지점 생성 (벽 장애물에서 충분히 떨어뜨리기)
    for i in range(num_agents):
        if i % 2 == 0:  # L->R
            target_x = np.random.uniform(corridor_width - 3.0, corridor_width - wall_margin)
        else:  # R->L
            target_x = np.random.uniform(wall_margin, 3.0)
        
        target_y = np.random.uniform(wall_margin, corridor_height - wall_margin)
        
        landmark = Landmark2D(id=i, x=target_x, y=target_y)
        landmarks.append(landmark)
    
    # 에이전트 생성 (벽 장애물에서 충분히 떨어뜨리기)
    for i in range(num_agents):
        if i % 2 == 0:  # L->R
            start_x = np.random.uniform(wall_margin, 3.0)
        else:  # R->L
            start_x = np.random.uniform(corridor_width - 3.0, corridor_width - wall_margin)
        
        start_y = np.random.uniform(wall_margin, corridor_height - wall_margin)
        max_speed = np.random.uniform(1.0, 2.0)
        
        agent = Agent2D(
            id=i, x=start_x, y=start_y, vx=0.0, vy=0.0,
            radius=agent_radius, target_id=i, max_speed=max_speed
        )
        agents.append(agent)
    
    return agents, landmarks


def create_obstacles(corridor_width: float, corridor_height: float,
                    bottleneck_position: float, bottleneck_width: float,
                    agent_radius: float) -> List[Obstacle2D]:
    """장애물 생성 - 환경의 모든 벽을 장애물 노드로 둘러싸기"""
    obstacles = []
    obstacle_id = 0
    
    # 에이전트 지름 = agent_radius * 2 = 1.0
    # 장애물 간격을 에이전트 지름의 1/3로 설정하여 촘촘하게
    obstacle_spacing = (agent_radius * 2) / 3  # ≈ 0.33
    obstacle_radius = obstacle_spacing / 2  # 장애물 반지름
    
    # 1. 상단 경계벽 (전체 너비)
    num_top_obstacles = int(corridor_width / obstacle_spacing) + 1
    for i in range(num_top_obstacles):
        x_pos = i * obstacle_spacing
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x_pos,
            y=corridor_height - obstacle_radius,  # 상단 경계
            radius=obstacle_radius
        )
        obstacles.append(obstacle)
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
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 3. 좌측 경계벽 (전체 높이)
    num_left_obstacles = int(corridor_height / obstacle_spacing) + 1
    for i in range(num_left_obstacles):
        y_pos = i * obstacle_spacing
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=obstacle_radius,  # 좌측 경계
            y=y_pos,
            radius=obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 4. 우측 경계벽 (전체 높이)
    for i in range(num_left_obstacles):
        y_pos = i * obstacle_spacing
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=corridor_width - obstacle_radius,  # 우측 경계
            y=y_pos,
            radius=obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 5. 병목 구간 벽들 - 통로는 완전히 열어둠
    center_y = corridor_height / 2
    wall_length = 2.0
    num_bottleneck_obstacles = int(wall_length / obstacle_spacing) + 1
    
    # 병목 통로의 상하 경계 계산 (여유 공간 추가)
    passage_margin = obstacle_radius * 2  # 통로 주변에 충분한 여유 공간
    passage_top = center_y + bottleneck_width/2 + passage_margin
    passage_bottom = center_y - bottleneck_width/2 - passage_margin
    
    # 병목 위쪽 벽
    upper_wall_y = center_y + bottleneck_width/2 + obstacle_radius
    for i in range(num_bottleneck_obstacles):
        x_pos = bottleneck_position - wall_length/2 + i * obstacle_spacing
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x_pos,
            y=upper_wall_y,
            radius=obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 병목 아래쪽 벽
    lower_wall_y = center_y - bottleneck_width/2 - obstacle_radius
    for i in range(num_bottleneck_obstacles):
        x_pos = bottleneck_position - wall_length/2 + i * obstacle_spacing
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x_pos,
            y=lower_wall_y,
            radius=obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 6. 병목 입구 좌측 벽 (좌측에서 병목까지, 통로 부분은 제외)
    left_wall_x = bottleneck_position - wall_length/2 - obstacle_radius
    
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
            obstacles.append(obstacle)
            obstacle_id += 1
    
    # 병목 통로 위에서 상단 경계까지
    start_y = passage_top + passage_margin  # 여유 공간 추가
    num_top_obstacles = int((corridor_height - start_y) / obstacle_spacing)
    for i in range(num_top_obstacles):
        y_pos = start_y + i * obstacle_spacing
        if y_pos < corridor_height - obstacle_radius:
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=left_wall_x,
                y=y_pos,
                radius=obstacle_radius
            )
            obstacles.append(obstacle)
            obstacle_id += 1
    
    # 7. 병목 입구 우측 벽 (병목에서 우측까지, 통로 부분은 제외)
    right_wall_x = bottleneck_position + wall_length/2 + obstacle_radius
    
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
            obstacles.append(obstacle)
            obstacle_id += 1
    
    # 병목 통로 위에서 상단 경계까지 (우측벽)
    for i in range(num_top_obstacles):
        y_pos = start_y + i * obstacle_spacing
        if y_pos < corridor_height - obstacle_radius:
            obstacle = Obstacle2D(
                id=obstacle_id,
                x=right_wall_x,
                y=y_pos,
                radius=obstacle_radius
            )
            obstacles.append(obstacle)
            obstacle_id += 1
    
    return obstacles