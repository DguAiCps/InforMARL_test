"""
Data types and structures for the bottleneck environment
"""
from dataclasses import dataclass
import math

# 엔티티 타입 매핑
ENTITY_TYPES = {"agent": 0, "landmark": 1, "obstacle": 2}


@dataclass
class Agent2D:
    """2D 원형 에이전트"""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    target_id: int  # 목표 landmark ID
    max_speed: float
    
    def get_distance_to(self, other_x: float, other_y: float) -> float:
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)
    
    def get_distance_to_agent(self, other: 'Agent2D') -> float:
        return self.get_distance_to(other.x, other.y)


@dataclass  
class Landmark2D:
    """2D 목표 지점"""
    id: int
    x: float
    y: float
    radius: float = 0.5


@dataclass
class Obstacle2D:
    """2D 장애물"""
    id: int
    x: float
    y: float
    radius: float