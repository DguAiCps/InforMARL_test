"""
Vectorized environment wrapper for parallel experience collection
"""
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import pickle
import traceback

from .bottleneck_env import BottleneckInforMARLEnv


def worker_process(worker_id: int, env_config: dict, command_queue: Queue, result_queue: Queue):
    """워커 프로세스 - 독립적으로 환경 실행하여 경험 수집"""
    try:
        # 워커별 독립 환경 생성 (시드는 환경에서 지원하지 않으므로 제거)
        env_config = env_config.copy()
        env_config.pop('seed', None)  # seed 파라미터 제거
        
        env = BottleneckInforMARLEnv(**env_config)
        observations = env.reset()
        
        print(f"Worker {worker_id} initialized")
        
        while True:
            try:
                command = command_queue.get(timeout=1.0)
                
                if command['type'] == 'step':
                    num_steps = command['num_steps']
                    experiences = []
                    
                    for step in range(num_steps):
                        # 환경 스텝 실행
                        observations, rewards, done, info = env.step()
                        
                        # 경험 저장 (단순화된 형태)
                        experience = {
                            'step': step,
                            'rewards': rewards,
                            'done': done,
                            'info': info,
                            'success_rate': info.get('success_rate', 0.0),
                            'timestep': env.timestep
                        }
                        experiences.append(experience)
                        
                        if done:
                            observations = env.reset()
                            break
                    
                    result_queue.put({
                        'worker_id': worker_id,
                        'experiences': experiences,
                        'final_info': info
                    })
                
                elif command['type'] == 'reset':
                    observations = env.reset()
                    result_queue.put({
                        'worker_id': worker_id,
                        'reset': True
                    })
                
                elif command['type'] == 'close':
                    break
                    
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                traceback.print_exc()
                result_queue.put({
                    'worker_id': worker_id,
                    'error': str(e)
                })
                break
    
    except Exception as e:
        print(f"Worker {worker_id} initialization error: {e}")
        traceback.print_exc()


class VectorizedBottleneckEnv:
    """병렬 환경 래퍼 - 여러 환경에서 동시에 경험 수집"""
    
    def __init__(self, env_config: dict, num_workers: int = 4):
        self.env_config = env_config
        self.num_workers = num_workers
        
        # 메인 환경 (학습용 네트워크 보유)
        self.main_env = BottleneckInforMARLEnv(**env_config)
        
        # 워커 프로세스 관련
        self.workers = []
        self.command_queues = []
        self.result_queue = Queue()
        
        self._start_workers()
    
    def _start_workers(self):
        """워커 프로세스들 시작"""
        print(f"Starting {self.num_workers} worker processes...")
        
        for worker_id in range(self.num_workers):
            command_queue = Queue()
            self.command_queues.append(command_queue)
            
            worker = Process(
                target=worker_process,
                args=(worker_id, self.env_config, command_queue, self.result_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"All {self.num_workers} workers started")
    
    def collect_parallel_experiences(self, steps_per_worker: int = 25) -> List[Dict]:
        """병렬로 경험 수집"""
        # 모든 워커에게 경험 수집 명령 전송
        for command_queue in self.command_queues:
            command_queue.put({
                'type': 'step',
                'num_steps': steps_per_worker
            })
        
        # 모든 워커 결과 수집
        all_experiences = []
        worker_infos = []
        
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=30.0)  # 30초 타임아웃
                
                if 'error' in result:
                    print(f"Worker {result['worker_id']} error: {result['error']}")
                    continue
                
                all_experiences.extend(result['experiences'])
                worker_infos.append({
                    'worker_id': result['worker_id'],
                    'final_info': result['final_info']
                })
                
            except Exception as e:
                print(f"Error collecting results: {e}")
        
        return all_experiences, worker_infos
    
    def reset_all_workers(self):
        """모든 워커 환경 리셋"""
        for command_queue in self.command_queues:
            command_queue.put({'type': 'reset'})
        
        # 리셋 완료 확인
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=10.0)
            except:
                pass
    
    def close(self):
        """모든 워커 프로세스 종료"""
        print("Closing vectorized environment...")
        
        for command_queue in self.command_queues:
            command_queue.put({'type': 'close'})
        
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        
        print("All workers closed")