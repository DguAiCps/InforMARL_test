"""
Rendering and visualization for bottleneck environment
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from typing import List, Optional
from ..utils.types import Agent2D, Landmark2D, Obstacle2D


class BottleneckRenderer:
    """병목 환경 렌더러"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def render(self, agents: List[Agent2D], landmarks: List[Landmark2D], obstacles: List[Obstacle2D],
               corridor_width: float, corridor_height: float, bottleneck_position: float, 
               bottleneck_width: float, timestep: int, success_count: int, collision_count: int,
               show_waypoints: bool = True):
        """환경 렌더링"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            plt.ion()  # 인터랙티브 모드
        
        self.ax.clear()
        
        # 환경 전체 배경
        self.ax.fill_between([0, corridor_width], 0, corridor_height, 
                            color='lightblue', alpha=0.2, label='복도')
        
        # 병목 구역 표시 (회색 벽들)
        center_y = corridor_height / 2
        bottleneck_x = bottleneck_position
        
        # 위쪽 벽
        upper_wall = patches.Rectangle(
            (bottleneck_x - 0.5, center_y + bottleneck_width/2), 
            1.0, corridor_height - (center_y + bottleneck_width/2),
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(upper_wall)
        
        # 아래쪽 벽  
        lower_wall = patches.Rectangle(
            (bottleneck_x - 0.5, 0), 
            1.0, center_y - bottleneck_width/2,
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(lower_wall)
        
        # 병목 통로 표시 (노란색으로 강조)
        bottleneck_passage = patches.Rectangle(
            (bottleneck_x - 0.5, center_y - bottleneck_width/2),
            1.0, bottleneck_width,
            facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2
        )
        self.ax.add_patch(bottleneck_passage)
        
        # 환경 경계 테두리
        boundary = patches.Rectangle(
            (0, 0), corridor_width, corridor_height,
            linewidth=3, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(boundary)
        
        # 병목 장애물 그리기 (원형 장애물들)
        for obstacle in obstacles:
            obs_circle = patches.Circle(
                (obstacle.x, obstacle.y), obstacle.radius,
                color='red', alpha=0.9, edgecolor='darkred', linewidth=2
            )
            self.ax.add_patch(obs_circle)
        
        # 목표 지점 그리기
        for i, landmark in enumerate(landmarks):
            goal_circle = patches.Circle(
                (landmark.x, landmark.y), landmark.radius,
                color='green', alpha=0.7, edgecolor='darkgreen', linewidth=2
            )
            self.ax.add_patch(goal_circle)
            self.ax.text(landmark.x, landmark.y, f'G{i}', 
                        ha='center', va='center', fontweight='bold', fontsize=10)
        
        # 에이전트 그리기
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for i, agent in enumerate(agents):
            color = colors[i % len(colors)]
            
            # 에이전트 원
            agent_circle = patches.Circle(
                (agent.x, agent.y), agent.radius,
                color=color, alpha=0.8, edgecolor='black', linewidth=1
            )
            self.ax.add_patch(agent_circle)
            
            # 에이전트 ID
            self.ax.text(agent.x, agent.y, str(i), 
                        ha='center', va='center', fontweight='bold', 
                        color='white', fontsize=8)
            
            # waypoint 경로 표시
            if show_waypoints and hasattr(agent, 'waypoints') and agent.waypoints:
                waypoints = agent.waypoints
                
                # waypoint들을 선으로 연결
                wp_x = [wp[0] for wp in waypoints]
                wp_y = [wp[1] for wp in waypoints]
                
                # 경로 선 그리기 (점선, 에이전트와 같은 색상)
                self.ax.plot(wp_x, wp_y, '--', color=color, alpha=0.6, linewidth=2, 
                           label=f'Agent {i} Path' if i < 3 else "")
                
                # waypoint 점들 표시
                for j, (wx, wy) in enumerate(waypoints[1:], 1):  # 시작점 제외
                    wp_circle = patches.Circle(
                        (wx, wy), 0.15, 
                        color=color, alpha=0.5, edgecolor=color, linewidth=1
                    )
                    self.ax.add_patch(wp_circle)
                    # waypoint 번호
                    self.ax.text(wx, wy, str(j), ha='center', va='center', 
                               fontsize=6, color='white', fontweight='bold')
                
                # 현재 목표 waypoint 강조
                if hasattr(agent, 'current_waypoint'):
                    cx, cy = agent.current_waypoint
                    current_wp = patches.Circle(
                        (cx, cy), 0.25, 
                        color=color, alpha=0.8, edgecolor='white', linewidth=2
                    )
                    self.ax.add_patch(current_wp)
            
            # 최종 목표까지의 선 (기존)
            target = landmarks[agent.target_id]
            self.ax.plot([agent.x, target.x], [agent.y, target.y], 
                        color=color, alpha=0.5, linestyle='--', linewidth=1.5)
            
            # 속도 벡터 (더 명확하게)
            speed = math.sqrt(agent.vx**2 + agent.vy**2)
            if speed > 0.1:
                scale = 3.0  # 화살표 크기 조정
                self.ax.arrow(agent.x, agent.y, agent.vx*scale, agent.vy*scale,
                            head_width=0.15, head_length=0.15, 
                            fc=color, ec=color, alpha=0.8, linewidth=2)
        
        # 설정
        self.ax.set_xlim(-0.5, corridor_width + 0.5)
        self.ax.set_ylim(-0.5, corridor_height + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'InforMARL 2D Bottleneck - Step {timestep}\\n성공: {success_count}, 충돌: {collision_count}', 
                         fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # 범례 추가
        legend_elements = [
            patches.Patch(color='darkgray', label='벽'),
            patches.Patch(color='yellow', alpha=0.3, label='병목 통로'),
            patches.Patch(color='green', label='목표'),
            patches.Patch(color='red', label='장애물')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # 강제로 화면 업데이트
        plt.draw()
        plt.pause(0.01)
        plt.show(block=False)