"""
InforMARL Bottleneck Environment - Main Entry Point
"""
import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.informarl_bneck.rl.runner import run_informarl_experiment, run_animation_demo, quick_movement_test
from src.informarl_bneck.utils.config import get_env_params, load_all_configs


def main():
    """메인 실행 함수"""
    # 🔥 YAML 파일에서 설정 읽기
    try:
        env_params = get_env_params("configs")
        configs = load_all_configs("configs")
        
        # YAML에서 읽은 값들
        num_agents = env_params['num_agents']
        num_episodes = configs.get('train', {}).get('training', {}).get('num_episodes', 100)
        
        # GPU 설정 읽기
        gpu_config = configs.get('gpu', {}).get('gpu', {})
        gpu_id = gpu_config.get('gpu_id', None)
        force_cpu = gpu_config.get('force_cpu', False)
        
        print("설정 로드됨:")
        print(f"   - 에이전트 수: {num_agents}")
        print(f"   - 에피소드: {num_episodes}")
        print(f"   - 병목 폭: {env_params['bottleneck_width']}")
        if gpu_id is not None:
            print(f"   - 지정 GPU: {gpu_id}")
        if force_cpu:
            print(f"   - CPU 강제 사용: {force_cpu}")
        
    except Exception as e:
        print(f"YAML 파일 로드 실패: {e}")
        print("기본값 사용...")
        num_agents = 4
        num_episodes = 100
        gpu_id = None
        force_cpu = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            # 애니메이션 데모 실행
            run_animation_demo(num_agents=num_agents)  # YAML 값 사용
        elif sys.argv[1] == "test":
            # 빠른 움직임 테스트  
            quick_movement_test(num_agents=min(num_agents, 2))  # YAML 값 사용 (최소 2개)
        else:
            print("사용법: python main.py [demo|test]")
    else:
        # 일반 학습 실행 (GPU 설정 포함)
        results, env = run_informarl_experiment(
            num_episodes=num_episodes, 
            num_agents=num_agents, 
            config=env_params,
            gpu_id=gpu_id,
            force_cpu=force_cpu
        )
        
        # 학습 후 애니메이션 보기
        print("\n학습 완료! 애니메이션으로 결과 확인 (y/n)?")
        if input().lower() == 'y':
            from src.informarl_bneck.rl.runner import evaluate_with_animation
            evaluate_with_animation(env, num_episodes=2, render_delay=0.15)


if __name__ == "__main__":
    main()