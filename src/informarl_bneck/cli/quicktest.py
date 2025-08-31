"""
Quick test CLI entry point
"""
import sys
from informarl_bneck.rl.runner import quick_movement_test


def main():
    """빠른 움직임 테스트"""
    if len(sys.argv) > 1:
        try:
            num_agents = int(sys.argv[1])
        except ValueError:
            num_agents = 2
    else:
        num_agents = 2
    
    print(f"빠른 테스트 시작: {num_agents} 에이전트")
    quick_movement_test(num_agents=num_agents)


if __name__ == "__main__":
    main()