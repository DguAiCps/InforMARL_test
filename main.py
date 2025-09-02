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
    # YAML 파일에서 설정 읽기
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
        # 일반 학습 실행 (GPU 설정 + 병렬 처리 포함)
        # performance.yaml 설정 추출
        perf_config = configs.get('performance', {}).get('performance', {})
        use_parallel = perf_config.get('use_parallel', True)
        num_workers = perf_config.get('num_workers', 4)
        
        print(f"\n성능 설정:")
        print(f"   - 병렬 처리: {use_parallel}")
        if use_parallel:
            print(f"   - 워커 수: {num_workers}")
            print(f"   - 워커별 스텝: {perf_config.get('steps_per_worker', 25)}")
        print(f"   - 메모리 정리: {perf_config.get('clear_memory_interval', 20)}에피소드마다")
        
        # train.yaml 설정 추출 및 표시
        train_config = configs.get('train', {})
        if train_config:
            print(f"\n훈련 설정:")
            training = train_config.get('training', {})
            if training:
                print(f"   - 에피소드: {training.get('num_episodes', 100)}")
                print(f"   - 배치 크기: {training.get('batch_size', 64)}")
                print(f"   - PPO 에폭: {training.get('ppo_epochs', 3)}")
                print(f"   - 학습률: {training.get('learning_rate', 0.003)}")
                print(f"   - 업데이트 빈도: {training.get('update_frequency', 25)}스텝마다")
                print(f"   - 평가 빈도: {training.get('eval_frequency', 5)}에피소드마다")
                print(f"   - GAE 감마: {training.get('gamma', 0.99)}")
                print(f"   - GAE 람다: {training.get('lambda', 0.95)}")
        
        # model.yaml 설정 추출 및 표시
        model_config = configs.get('model', {})
        if model_config:
            print(f"\n모델 설정:")
            gnn = model_config.get('gnn', {})
            if gnn:
                print(f"   - GNN 히든: {gnn.get('hidden_dim', 64)}")
                print(f"   - GNN 레이어: {gnn.get('num_layers', 1)}")
                print(f"   - GNN 임베딩: {gnn.get('num_embeddings', 4)}개")
            actor = model_config.get('actor', {})
            critic = model_config.get('critic', {})
            if actor:
                print(f"   - Actor 히든: {actor.get('hidden_dim', 64)}")
            if critic:
                print(f"   - Critic 히든: {critic.get('hidden_dim', 64)}")
        
        results, env = run_informarl_experiment(
            num_episodes=num_episodes, 
            num_agents=num_agents, 
            config={**env_params, 'model': model_config},  # env + model 설정 전달
            train_config=train_config,  # train.yaml 전달
            gpu_id=gpu_id,
            force_cpu=force_cpu,
            use_parallel=use_parallel,
            num_workers=num_workers
        )
        
        # 학습 후 애니메이션 보기
        print("\n학습 완료! 애니메이션으로 결과 확인 (y/n)?")
        if input().lower() == 'y':
            from src.informarl_bneck.rl.runner import evaluate_with_animation
            evaluate_with_animation(env, num_episodes=2, render_delay=0.15)


if __name__ == "__main__":
    main()