"""
Demo CLI entry point for animation
"""
import sys
from ..rl.runner import run_animation_demo


def main():
    """애니메이션 데모 실행"""
    if len(sys.argv) > 1:
        try:
            num_agents = int(sys.argv[1])
        except ValueError:
            num_agents = 4
    else:
        num_agents = 4
    
    print(f"애니메이션 데모 시작: {num_agents} 에이전트")
    run_animation_demo(num_agents=num_agents)


if __name__ == "__main__":
    main()