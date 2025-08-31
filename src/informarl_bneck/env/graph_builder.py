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