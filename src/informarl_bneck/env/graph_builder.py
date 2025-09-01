"""
Graph construction for InforMARL observations
"""
import torch
import math
from typing import List
from torch_geometric.data import Data

from ..utils.types import Agent2D, Landmark2D, Obstacle2D, ENTITY_TYPES


def build_graph_observations(agents: List[Agent2D], landmarks: List[Landmark2D], 
                           obstacles: List[Obstacle2D], sensing_radius: float) -> List[Data]:
    """InforMARL 방식 그래프 관측 생성"""
    observations = []
    
    # 각 에이전트마다 자신을 기준으로 한 그래프 생성
    for ego_agent in agents:
        node_features = []
        entity_types = []
        all_entities = []
        
        # 1. 에이전트 노드들
        for agent in agents:
            # 상대 위치/속도/목표 계산 (sensing_radius로 정규화)
            rel_x = (agent.x - ego_agent.x) / sensing_radius
            rel_y = (agent.y - ego_agent.y) / sensing_radius
            rel_vx = (agent.vx - ego_agent.vx) / agent.max_speed
            rel_vy = (agent.vy - ego_agent.vy) / agent.max_speed
            
            # 목표 위치
            target = landmarks[agent.target_id]
            rel_goal_x = (target.x - ego_agent.x) / sensing_radius
            rel_goal_y = (target.y - ego_agent.y) / sensing_radius
            
            features = [rel_x, rel_y, rel_vx, rel_vy, rel_goal_x, rel_goal_y]
            node_features.append(features)
            entity_types.append(ENTITY_TYPES["agent"])
            all_entities.append(('agent', agent))
        
        # 2. 목표 지점 노드들
        for landmark in landmarks:
            rel_x = (landmark.x - ego_agent.x) / sensing_radius
            rel_y = (landmark.y - ego_agent.y) / sensing_radius
            
            features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # 목표는 정지, 목표=자기위치
            node_features.append(features)
            entity_types.append(ENTITY_TYPES["landmark"])
            all_entities.append(('landmark', landmark))
        
        # 3. 장애물 노드들
        for obstacle in obstacles:
            rel_x = (obstacle.x - ego_agent.x) / sensing_radius
            rel_y = (obstacle.y - ego_agent.y) / sensing_radius
            
            features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # 장애물은 정지
            node_features.append(features)
            entity_types.append(ENTITY_TYPES["obstacle"])
            all_entities.append(('obstacle', obstacle))
        
        # 4. 엣지 생성 (논문의 방향성 규칙에 따라)
        edge_index, edge_attr = build_edges(all_entities, ego_agent, sensing_radius)
        
        # 그래프 데이터 생성
        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            entity_type=torch.tensor(entity_types, dtype=torch.long)
        )
        
        observations.append(graph_data)
    
    return observations


def build_edges(all_entities, ego_agent: Agent2D, sensing_radius: float):
    """엣지 생성 - InforMARL 방향성 규칙"""
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
                if dist_i <= sensing_radius and dist_j <= sensing_radius:
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
                        
                        edge_attr.append([edge_dist / sensing_radius])  # 정규화
    
    # 최소 연결 보장
    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0.0]]
    
    return edge_index, edge_attr


# =============================================================================
# 🚀 GPU 배치 그래프 구축 함수들
# =============================================================================

def batch_build_graph_observations_gpu(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                      obstacles: List[Obstacle2D], sensing_radius: float, 
                                      device: torch.device) -> List[Data]:
    """GPU에서 모든 에이전트의 그래프 관측을 배치로 생성 - 대폭 성능 향상"""
    
    num_agents = len(agents)
    num_landmarks = len(landmarks)
    num_obstacles = len(obstacles)
    total_entities = num_agents + num_landmarks + num_obstacles
    
    if total_entities == 0:
        return []
    
    # 🚀 모든 위치 데이터를 GPU로 한번에 전송
    agent_positions = torch.tensor([[a.x, a.y] for a in agents], dtype=torch.float32).to(device)
    agent_velocities = torch.tensor([[a.vx, a.vy] for a in agents], dtype=torch.float32).to(device)
    agent_max_speeds = torch.tensor([a.max_speed for a in agents], dtype=torch.float32).to(device)
    
    landmark_positions = torch.tensor([[l.x, l.y] for l in landmarks], dtype=torch.float32).to(device)
    obstacle_positions = torch.tensor([[o.x, o.y] for o in obstacles], dtype=torch.float32).to(device)
    
    # 목표 위치 (각 에이전트의 target_id에 해당하는 landmark)
    target_positions = torch.tensor([[landmarks[a.target_id].x, landmarks[a.target_id].y] for a in agents], dtype=torch.float32).to(device)
    
    # 모든 엔티티 위치 결합
    all_positions = torch.cat([agent_positions, landmark_positions, obstacle_positions], dim=0)
    
    # 🚀 모든 거리 계산을 GPU에서 배치로
    # [num_agents, total_entities] 크기의 거리 행렬
    ego_positions_expanded = agent_positions.unsqueeze(1)  # [num_agents, 1, 2]
    all_positions_expanded = all_positions.unsqueeze(0)    # [1, total_entities, 2]
    
    # 모든 ego agent로부터 모든 엔티티까지의 거리 (배치)
    distance_matrix = torch.norm(ego_positions_expanded - all_positions_expanded, dim=2)  # [num_agents, total_entities]
    
    # 센싱 반경 내 엔티티 마스크
    sensing_mask = distance_matrix <= sensing_radius  # [num_agents, total_entities]
    
    observations = []
    
    for ego_idx in range(num_agents):
        ego_agent = agents[ego_idx]
        ego_pos = agent_positions[ego_idx]
        ego_vel = agent_velocities[ego_idx]
        ego_target_pos = target_positions[ego_idx]
        
        # 이 ego agent가 센싱할 수 있는 엔티티들
        visible_mask = sensing_mask[ego_idx]  # [total_entities]
        visible_indices = torch.where(visible_mask)[0]
        
        if len(visible_indices) == 0:
            # 아무것도 보이지 않으면 자신만 포함
            visible_indices = torch.tensor([ego_idx], device=device)
        
        # 🚀 GPU에서 노드 특성 배치 계산
        node_features, entity_types = batch_compute_node_features_gpu(
            ego_pos, ego_vel, ego_target_pos, 
            all_positions, visible_indices, 
            num_agents, num_landmarks, num_obstacles,
            agent_velocities, target_positions, agent_max_speeds,
            sensing_radius, device
        )
        
        # 🚀 GPU에서 엣지 배치 계산
        edge_index, edge_attr = batch_compute_edges_gpu(
            all_positions, visible_indices, ego_pos,
            num_agents, num_landmarks, num_obstacles,
            sensing_radius, device
        )
        
        # 그래프 데이터 생성
        graph_data = Data(
            x=node_features.cpu(),
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(),
            entity_type=entity_types.cpu()
        )
        
        observations.append(graph_data)
    
    return observations


def batch_compute_node_features_gpu(ego_pos, ego_vel, ego_target_pos, 
                                   all_positions, visible_indices,
                                   num_agents, num_landmarks, num_obstacles,
                                   agent_velocities, target_positions, agent_max_speeds,
                                   sensing_radius, device):
    """GPU에서 노드 특성 배치 계산"""
    
    num_visible = len(visible_indices)
    visible_positions = all_positions[visible_indices]  # [num_visible, 2]
    
    # 상대 위치 (배치)
    rel_positions = (visible_positions - ego_pos) / sensing_radius  # [num_visible, 2]
    
    # 각 엔티티 타입별 특성 계산
    node_features = []
    entity_types = []
    
    for i, global_idx in enumerate(visible_indices):
        global_idx_int = global_idx.item()
        
        if global_idx_int < num_agents:
            # 에이전트
            agent_idx = global_idx_int
            rel_vel = (agent_velocities[agent_idx] - ego_vel) / agent_max_speeds[agent_idx]
            rel_goal = (target_positions[agent_idx] - ego_pos) / sensing_radius
            
            features = torch.cat([
                rel_positions[i],  # rel_x, rel_y
                rel_vel,          # rel_vx, rel_vy
                rel_goal          # rel_goal_x, rel_goal_y
            ])
            entity_types.append(ENTITY_TYPES["agent"])
            
        elif global_idx_int < num_agents + num_landmarks:
            # 목표점
            rel_goal = rel_positions[i]  # 목표점의 목표는 자기 자신
            features = torch.cat([
                rel_positions[i],           # rel_x, rel_y
                torch.zeros(2, device=device),  # 속도 0
                rel_goal                    # rel_goal_x, rel_goal_y
            ])
            entity_types.append(ENTITY_TYPES["landmark"])
            
        else:
            # 장애물
            features = torch.cat([
                rel_positions[i],           # rel_x, rel_y
                torch.zeros(2, device=device),  # 속도 0
                rel_positions[i]            # 장애물의 "목표"는 자기 위치
            ])
            entity_types.append(ENTITY_TYPES["obstacle"])
        
        node_features.append(features)
    
    node_features = torch.stack(node_features) if node_features else torch.zeros((1, 6), device=device)
    entity_types = torch.tensor(entity_types, dtype=torch.long, device=device)
    
    return node_features, entity_types


def batch_compute_edges_gpu(all_positions, visible_indices, ego_pos,
                           num_agents, num_landmarks, num_obstacles,
                           sensing_radius, device):
    """GPU에서 엣지 배치 계산"""
    
    num_visible = len(visible_indices)
    visible_positions = all_positions[visible_indices]
    
    if num_visible <= 1:
        # 자기 자신에게만 연결
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        return edge_index, edge_attr
    
    # 🚀 모든 visible 엔티티 간 거리 행렬 (배치)
    distance_matrix = torch.cdist(visible_positions, visible_positions, p=2)  # [num_visible, num_visible]
    
    edge_indices = []
    edge_attributes = []
    
    for i in range(num_visible):
        for j in range(num_visible):
            if i != j:
                global_i = visible_indices[i].item()
                global_j = visible_indices[j].item()
                
                # 엔티티 타입 결정
                type_i = get_entity_type_from_global_idx(global_i, num_agents, num_landmarks, num_obstacles)
                type_j = get_entity_type_from_global_idx(global_j, num_agents, num_landmarks, num_obstacles)
                
                # InforMARL 연결 규칙
                should_connect = False
                if type_i == 'agent' and type_j == 'agent':
                    should_connect = True
                elif type_i in ['landmark', 'obstacle'] and type_j == 'agent':
                    should_connect = True
                
                if should_connect:
                    edge_indices.append([i, j])
                    edge_dist = distance_matrix[i, j] / sensing_radius  # 정규화
                    edge_attributes.append([edge_dist.item()])
    
    if not edge_indices:
        edge_indices = [[0, 0]]
        edge_attributes = [[0.0]]
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float32, device=device)
    
    return edge_index, edge_attr


def get_entity_type_from_global_idx(global_idx, num_agents, num_landmarks, num_obstacles):
    """글로벌 인덱스로부터 엔티티 타입 결정"""
    if global_idx < num_agents:
        return 'agent'
    elif global_idx < num_agents + num_landmarks:
        return 'landmark'
    else:
        return 'obstacle'