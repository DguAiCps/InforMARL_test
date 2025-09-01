"""
Training and evaluation runners for InforMARL
"""
import numpy as np
import time
from typing import List, Dict, Any

from ..env import BottleneckInforMARLEnv
from ..env.graph_builder import build_graph_observations
from ..utils.device import clear_gpu_memory, get_memory_usage


def train_agents(env: BottleneckInforMARLEnv, num_episodes: int = 100) -> List[float]:
    """실제 학습이 포함된 함수"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(env.max_timesteps):
            observations, rewards, done, info = env.step()
            episode_reward += sum(rewards)
            
            # 매 N스텝마다 네트워크 업데이트 (성능 최적화: 10 → 50)
            if step % 50 == 0:
                for agent in env.informarl_agents:
                    agent.update_networks(env.shared_gnn)
                # 공유 GNN 옵티마이저 스텝
                env.gnn_optimizer.step()
                env.gnn_optimizer.zero_grad()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 에피소드 끝에 한 번 더 업데이트
        for agent in env.informarl_agents:
            agent.update_networks(env.shared_gnn)
        # 공유 GNN 옵티마이저 스텝
        env.gnn_optimizer.step()
        env.gnn_optimizer.zero_grad()
            
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            memory_info = get_memory_usage()
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {info['success_rate']:.2f}, Memory: {memory_info}")
        
        # 🚀 서버 GPU 메모리 관리: 매 20 에피소드마다 캐시 정리
        if episode % 20 == 0 and episode > 0:
            clear_gpu_memory()
    
    return episode_rewards


def evaluate_with_animation(env: BottleneckInforMARLEnv, num_episodes: int = 5, 
                          render_delay: float = 0.2) -> List[float]:
    """평가 모드로 에이전트 실행하며 애니메이션 표시"""
    print("=== InforMARL 평가 모드 (애니메이션) ===")
    print("창이 열리면 에이전트 움직임을 관찰하세요!")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n에피소드 {episode + 1}/{num_episodes} 시작")
        observations = env.reset()
        episode_reward = 0
        
        # 초기 상태 렌더링
        env.render()
        time.sleep(render_delay * 2)  # 초기 상태 좀 더 오래 보여주기
        
        for step in range(env.max_timesteps):
            # 평가 모드로 행동 선택 (training=False)
            actions, _, _ = env._get_batch_actions(
                build_graph_observations(env.agents, env.landmarks, env.obstacles, env.sensing_radius), 
                training=False
            )
            
            # 한 스텝 실행
            observations, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
            
            # 렌더링
            env.render()
            
            # 움직임 확인을 위한 디버그 출력
            if step % 20 == 0:
                print(f"    스텝 {step}: 에이전트 위치들")
                for i, agent in enumerate(env.agents):
                    print(f"      Agent {i}: ({agent.x:.1f}, {agent.y:.1f}) 속도: ({agent.vx:.2f}, {agent.vy:.2f})")
            
            time.sleep(render_delay)
            
            if done:
                print(f"  에피소드 완료! 스텝: {step + 1}")
                break
        
        episode_rewards.append(episode_reward)
        print(f"  에피소드 보상: {episode_reward:.2f}")
        print(f"  성공률: {info['success_rate']:.2f}")
        print(f"  충돌 횟수: {info['collision_count']}")
        
        # 에피소드 간 잠시 대기
        print("  다음 에피소드까지 잠시 대기...")
        time.sleep(2.0)
    
    avg_reward = np.mean(episode_rewards)
    print(f"\n=== 평가 결과 ===")
    print(f"평균 에피소드 보상: {avg_reward:.3f}")
    
    return episode_rewards


def run_informarl_experiment(num_episodes: int = 100, num_agents: int = 6, config: dict = None, gpu_id: int = None, force_cpu: bool = False) -> tuple[List[float], BottleneckInforMARLEnv]:
    """InforMARL 실험 실행"""
    print("=== InforMARL 2D 병목 환경 학습 시작 ===")
    
    # 🔥 설정 및 GPU 옵션을 환경에 전달
    env = BottleneckInforMARLEnv(num_agents=num_agents, config=config, gpu_id=gpu_id, force_cpu=force_cpu)
    episode_rewards = train_agents(env, num_episodes=num_episodes)
    
    print(f"\n=== 최종 결과 ===")
    print(f"평균 에피소드 보상: {np.mean(episode_rewards):.3f}")
    
    return episode_rewards, env


def run_animation_demo(num_agents: int = 4) -> List[float]:
    """애니메이션 데모 실행"""
    print("=== InforMARL 애니메이션 데모 ===")
    
    env = BottleneckInforMARLEnv(num_agents=num_agents)
    
    # 간단한 학습 (선택사항)
    print("간단한 사전 학습 중...")
    train_agents(env, num_episodes=10)
    
    # 애니메이션으로 평가
    print("\n평가 모드 애니메이션 시작!")
    results = evaluate_with_animation(env, num_episodes=3, render_delay=0.2)
    
    return results


def quick_movement_test(num_agents: int = 2) -> bool:
    """에이전트 움직임 빠른 테스트"""
    print("=== 에이전트 움직임 테스트 ===")
    
    env = BottleneckInforMARLEnv(num_agents=num_agents, max_timesteps=50)
    observations = env.reset()
    
    print("초기 상태 렌더링...")
    env.render()
    time.sleep(1)
    
    for step in range(20):
        # 랜덤 행동으로 테스트
        random_actions = [np.random.randint(0, 4) for _ in range(num_agents)]
        
        print(f"스텝 {step}: 행동 {random_actions}")
        observations, rewards, done, info = env.step(random_actions)
        
        # 에이전트 위치 출력
        for i, agent in enumerate(env.agents):
            print(f"  Agent {i}: ({agent.x:.2f}, {agent.y:.2f}) 속도: ({agent.vx:.2f}, {agent.vy:.2f})")
        
        env.render()
        time.sleep(0.5)
        
        if done:
            break
    
    print("테스트 완료!")
    import matplotlib.pyplot as plt
    plt.show()
    return True