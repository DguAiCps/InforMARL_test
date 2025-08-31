"""
Training CLI entry point
"""
import sys
from informarl_bneck.env import BottleneckInforMARLEnv
from informarl_bneck.rl.runner import run_informarl_experiment


def main():
    """메인 학습 실행"""
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            num_episodes = 100
    else:
        num_episodes = 100
    
    if len(sys.argv) > 2:
        try:
            num_agents = int(sys.argv[2])
        except ValueError:
            num_agents = 4
    else:
        num_agents = 4
    
    print(f"학습 시작: {num_episodes} 에피소드, {num_agents} 에이전트")
    
    # 일반 학습 실행
    results, env = run_informarl_experiment(num_episodes=num_episodes, num_agents=num_agents)
    
    # 학습 후 애니메이션 보기
    print("\n학습 완료! 애니메이션으로 결과 확인 (y/n)?")
    if input().lower() == 'y':
        from informarl_bneck.rl.runner import evaluate_with_animation
        evaluate_with_animation(env, num_episodes=2, render_delay=0.15)


if __name__ == "__main__":
    main()