"""
Training and evaluation runners for InforMARL
"""
import numpy as np
import time
from typing import List, Dict, Any

from ..env import BottleneckInforMARLEnv
from ..env.graph_builder import build_graph_observations
from ..env.vec_env import VectorizedBottleneckEnv
from ..utils.device import clear_gpu_memory, get_memory_usage


def train_agents_parallel(env: BottleneckInforMARLEnv, vec_env: VectorizedBottleneckEnv, 
                         num_episodes: int = 100) -> List[float]:
    """병렬 경험 수집을 활용한 학습 함수"""
    episode_rewards = []
    total_steps = 0
    # performance.yaml에서 steps_per_worker 사용
    steps_per_collection = 25  # 기본값
    # 일단 기본값 사용 (추후 performance 설정 전달 가능)
    
    print(f"=== 병렬 학습 시작: {vec_env.num_workers}개 워커 사용 ===")
    
    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # 🚀 병렬로 경험 수집 (4배 빠름!)
            print(f"Episode {episode}: 병렬 경험 수집 시작...")
            all_experiences, worker_infos = vec_env.collect_parallel_experiences(steps_per_collection)
            
            # 수집된 경험 통계 수정
            total_rewards = 0
            for exp in all_experiences:
                rewards = exp['rewards']
                if isinstance(rewards, list):
                    total_rewards += sum(rewards)
                else:
                    total_rewards += rewards if rewards else 0
            
            print(f"  디버그: 전체 경험 {len(all_experiences)}개, 총 보상 {total_rewards:.2f}")
            if len(all_experiences) > 0:
                print(f"  디버그: 첫 번째 경험 보상: {all_experiences[0]['rewards']}")
            
            episode_rewards.append(total_rewards)
            total_steps += len(all_experiences)
            
            # 🚀 수집된 경험으로 네트워크 학습 (메인 환경 사용)
            print(f"  수집된 경험: {len(all_experiences)}개, 보상합: {total_rewards:.2f}")
            
            # 디버깅: 보상이 0이면 원인 추적
            if total_rewards == 0 and len(all_experiences) > 0:
                print(f"  경고: 보상이 0입니다. 첫 번째 경험 보상: {all_experiences[0]['rewards']}")
                print(f"  첫 번째 경험 info: {all_experiences[0]['info']}")
            # 네트워크 업데이트는 보상이 있을 때만 수행
            if total_rewards != 0:
                env._update_shared_networks()
                if hasattr(env, 'gnn_optimizer') and env.gnn_optimizer is not None:
                    env.gnn_optimizer.step()
                    env.gnn_optimizer.zero_grad()
            else:
                print("  보상이 0이므로 네트워크 업데이트 스킵")
            
            # train.yaml에서 eval_frequency 사용
            eval_freq = 5  # 기본값
            if hasattr(env, 'train_config') and 'training' in env.train_config:
                eval_freq = env.train_config['training'].get('eval_frequency', 5)
            
            if episode % eval_freq == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                episode_time = time.time() - episode_start_time
                
                # 워커별 성공률 수집
                success_rates = [info['final_info'].get('success_rate', 0.0) for info in worker_infos]
                avg_success_rate = np.mean(success_rates) if success_rates else 0.0
                
                memory_info = get_memory_usage()
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Success Rate = {avg_success_rate:.2f}, "
                      f"Time = {episode_time:.2f}s, Memory: {memory_info}")
            
            # performance.yaml에서 clear_memory_interval 사용
            clear_interval = 20  # 기본값
            # 일단 우선 기본값 사용 (추후 개선 가능)
            if episode % clear_interval == 0 and episode > 0:
                clear_gpu_memory()
    
    except KeyboardInterrupt:
        print("\n학습 중단됨")
    finally:
        vec_env.close()
    
    return episode_rewards


def train_agents(env: BottleneckInforMARLEnv, num_episodes: int = 100) -> List[float]:
    """기존 단일 환경 학습 함수 (호환성 유지)"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(env.max_timesteps):
            observations, rewards, done, info = env.step()
            episode_reward += sum(rewards)
            
            # train.yaml에서 update_frequency 사용
            update_freq = 25  # 기본값
            if hasattr(env, 'train_config') and 'training' in env.train_config:
                update_freq = env.train_config['training'].get('update_frequency', 25)
            
            if step % update_freq == 0:
                # 🚀 공유 네트워크 배치 업데이트 (개별 업데이트 대신)
                env._update_shared_networks()
                # 공유 GNN 옵티마이저 스텝
                if hasattr(env, 'gnn_optimizer') and env.gnn_optimizer is not None:
                    env.gnn_optimizer.step()
                    env.gnn_optimizer.zero_grad()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 에피소드 끝에 한 번 더 업데이트
        env._update_shared_networks()
        # 공유 GNN 옵티마이저 스텝
        if hasattr(env, 'gnn_optimizer') and env.gnn_optimizer is not None:
            env.gnn_optimizer.step()
            env.gnn_optimizer.zero_grad()
            
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            memory_info = get_memory_usage()
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {info['success_rate']:.2f}, Memory: {memory_info}")
        
        # train.yaml에서 설정된 빈도로 GPU 메모리 정리  
        clear_interval = 20  # 기본값
        if episode % clear_interval == 0 and episode > 0:
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


def run_informarl_experiment(num_episodes: int = 100, num_agents: int = 6, config: dict = None, train_config: dict = None, gpu_id: int = None, force_cpu: bool = False, use_parallel: bool = True, num_workers: int = 4) -> tuple[List[float], BottleneckInforMARLEnv]:
    """InforMARL 실험 실행 (병렬 처리 옵션 추가)"""
    print("=== InforMARL 2D 병목 환경 학습 시작 ===")
    
    # 메인 환경 생성 (학습용 네트워크 보유)
    main_env = BottleneckInforMARLEnv(num_agents=num_agents, config=config, train_config=train_config, gpu_id=gpu_id, force_cpu=force_cpu)
    
    if use_parallel and num_workers > 1:
        print(f"\n🚀 병렬 모드: {num_workers}개 워커 사용")
        
        # 병렬 환경 설정
        env_config = {
            'num_agents': num_agents,
            'config': config,
            'train_config': train_config,  # train.yaml 설정 전달
            'gpu_id': None,  # 워커는 CPU 사용
            'force_cpu': True  # 워커는 CPU로 강제
        }
        
        vec_env = VectorizedBottleneckEnv(env_config, num_workers=num_workers)
        episode_rewards = train_agents_parallel(main_env, vec_env, num_episodes=num_episodes)
    else:
        print("\n일반 모드: 단일 환경 사용")
        episode_rewards = train_agents(main_env, num_episodes=num_episodes)
    
    print(f"\n=== 최종 결과 ===")
    print(f"평균 에피소드 보상: {np.mean(episode_rewards):.3f}")
    
    return episode_rewards, main_env


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
    
    for step in range(300):
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