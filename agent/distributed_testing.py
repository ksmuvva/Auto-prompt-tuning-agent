"""
Distributed Prompt Testing Framework

This module provides distributed testing capabilities:
1. Parallel execution across multiple workers
2. Task queue with priority scheduling
3. Load balancing
4. Result aggregation
5. Fault tolerance and retry logic
6. Progress tracking

Program of Thoughts:
1. Create task queue with prompt test jobs
2. Spawn worker processes/threads to execute tests
3. Distribute tasks across workers with load balancing
4. Collect results as they complete
5. Aggregate results and handle failures
6. Provide progress monitoring
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
from enum import Enum
import time
import threading
import traceback
from functools import partial


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Represents a single test task"""
    task_id: str
    prompt_id: str
    test_function: str  # Serializable function name
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value

    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'prompt_id': self.prompt_id,
            'status': self.status.value,
            'priority': self.priority.value,
            'retry_count': self.retry_count,
            'duration': self.duration,
            'result': self.result,
            'error': self.error
        }


@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: int
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time: float = 0.0
    avg_task_time: float = 0.0
    current_task: Optional[str] = None
    is_busy: bool = False


class DistributedTestExecutor:
    """
    Distributed test executor with parallel workers

    Manages task distribution, execution, and result aggregation.
    """

    def __init__(self,
                 num_workers: int = None,
                 use_processes: bool = False,
                 max_retries: int = 3):
        """
        Initialize distributed executor

        Args:
            num_workers: Number of workers (defaults to CPU count)
            use_processes: Use processes instead of threads (for CPU-bound tasks)
            max_retries: Maximum retry attempts for failed tasks
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.max_retries = max_retries

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: PriorityQueue = PriorityQueue()
        self.results: Dict[str, Any] = {}

        # Worker management
        self.worker_stats: Dict[int, WorkerStats] = {
            i: WorkerStats(worker_id=i) for i in range(self.num_workers)
        }

        # Synchronization
        self.lock = threading.Lock()
        self.running = False

        # Executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit_task(self, task: Task):
        """
        Submit a task for execution

        Args:
            task: Task to execute
        """
        with self.lock:
            self.tasks[task.task_id] = task
            self.task_queue.put((task.priority.value, task.task_id, task))

    def submit_batch(self, tasks: List[Task]):
        """Submit multiple tasks"""
        for task in tasks:
            self.submit_task(task)

    def _execute_task(self, task: Task, worker_id: int) -> Tuple[str, Any, Optional[str]]:
        """
        Execute a single task

        Args:
            task: Task to execute
            worker_id: Worker executing the task

        Returns:
            Tuple of (task_id, result, error)
        """
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        # Update worker stats
        with self.lock:
            self.worker_stats[worker_id].is_busy = True
            self.worker_stats[worker_id].current_task = task.task_id

        try:
            # Import and execute the test function
            # This assumes the function is importable and serializable
            module_path, func_name = task.test_function.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            test_func = getattr(module, func_name)

            # Execute with timeout protection
            result = test_func(*task.args, **task.kwargs)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = time.time()

            # Update worker stats
            with self.lock:
                stats = self.worker_stats[worker_id]
                stats.tasks_completed += 1
                stats.total_time += task.duration
                stats.avg_task_time = stats.total_time / stats.tasks_completed
                stats.is_busy = False
                stats.current_task = None

            return task.task_id, result, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            task.error = error_msg
            task.end_time = time.time()

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                # Resubmit with lower priority
                task.priority = TaskPriority.LOW
                self.submit_task(task)
            else:
                task.status = TaskStatus.FAILED

                with self.lock:
                    self.worker_stats[worker_id].tasks_failed += 1
                    self.worker_stats[worker_id].is_busy = False
                    self.worker_stats[worker_id].current_task = None

            return task.task_id, None, error_msg

    def execute_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute all submitted tasks

        Args:
            timeout: Maximum time to wait for all tasks (seconds)

        Returns:
            Dictionary mapping task_id to results
        """
        self.running = True
        start_time = time.time()

        # Create futures for all tasks
        futures = {}
        worker_assignments = {}
        worker_id = 0

        # Submit tasks to executor
        with self.lock:
            task_items = list(self.tasks.values())

        for task in task_items:
            if task.status == TaskStatus.PENDING:
                future = self.executor.submit(
                    self._execute_task,
                    task,
                    worker_id % self.num_workers
                )
                futures[future] = task.task_id
                worker_assignments[task.task_id] = worker_id % self.num_workers
                worker_id += 1

        # Collect results as they complete
        for future in as_completed(futures, timeout=timeout):
            task_id = futures[future]

            try:
                result_task_id, result, error = future.result()

                with self.lock:
                    if error is None:
                        self.results[result_task_id] = result
                    else:
                        self.results[result_task_id] = {'error': error}

            except Exception as e:
                with self.lock:
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].error = str(e)
                    self.results[task_id] = {'error': str(e)}

        self.running = False
        return self.results

    def execute_parallel_tests(self,
                               test_configs: List[Dict],
                               test_function_name: str) -> List[Dict]:
        """
        Execute prompt tests in parallel

        Program of Thoughts:
        1. Convert test configs to Task objects
        2. Submit all tasks
        3. Execute in parallel across workers
        4. Aggregate results
        5. Return sorted by performance

        Args:
            test_configs: List of test configurations
            test_function_name: Full path to test function (e.g., 'agent.test_prompt')

        Returns:
            List of test results
        """
        # Create tasks
        tasks = []
        for i, config in enumerate(test_configs):
            task = Task(
                task_id=f"test_{i}",
                prompt_id=config.get('prompt_id', f'prompt_{i}'),
                test_function=test_function_name,
                kwargs=config,
                priority=TaskPriority.NORMAL,
                max_retries=self.max_retries
            )
            tasks.append(task)

        # Submit and execute
        self.submit_batch(tasks)
        results = self.execute_all()

        # Aggregate results
        aggregated = []
        for task_id, result in results.items():
            task = self.tasks[task_id]
            aggregated.append({
                'task_id': task_id,
                'prompt_id': task.prompt_id,
                'status': task.status.value,
                'result': result,
                'duration': task.duration,
                'retry_count': task.retry_count
            })

        return aggregated

    def get_progress(self) -> Dict:
        """
        Get current execution progress

        Returns:
            Progress statistics
        """
        with self.lock:
            total_tasks = len(self.tasks)
            completed = sum(1 for t in self.tasks.values()
                          if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values()
                        if t.status == TaskStatus.FAILED)
            running = sum(1 for t in self.tasks.values()
                         if t.status == TaskStatus.RUNNING)
            pending = total_tasks - completed - failed - running

            return {
                'total_tasks': total_tasks,
                'completed': completed,
                'failed': failed,
                'running': running,
                'pending': pending,
                'progress_pct': (completed / total_tasks * 100) if total_tasks > 0 else 0,
                'worker_stats': {
                    wid: {
                        'tasks_completed': stats.tasks_completed,
                        'tasks_failed': stats.tasks_failed,
                        'avg_task_time': stats.avg_task_time,
                        'is_busy': stats.is_busy,
                        'current_task': stats.current_task
                    }
                    for wid, stats in self.worker_stats.items()
                }
            }

    def get_summary(self) -> Dict:
        """Get execution summary"""
        progress = self.get_progress()

        total_duration = sum(
            t.duration for t in self.tasks.values()
            if t.duration is not None
        )
        avg_duration = (
            total_duration / progress['completed']
            if progress['completed'] > 0 else 0
        )

        return {
            **progress,
            'total_duration': total_duration,
            'avg_task_duration': avg_duration,
            'success_rate': (
                progress['completed'] / progress['total_tasks']
                if progress['total_tasks'] > 0 else 0
            ),
            'tasks': [t.to_dict() for t in self.tasks.values()]
        }

    def shutdown(self):
        """Shutdown executor"""
        self.running = False
        self.executor.shutdown(wait=True)


class LoadBalancer:
    """
    Load balancer for distributing tasks across workers

    Implements various load balancing strategies.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_loads: Dict[int, int] = {i: 0 for i in range(num_workers)}

    def assign_task(self, strategy: str = "least_loaded") -> int:
        """
        Assign task to a worker using specified strategy

        Strategies:
        - round_robin: Simple round-robin
        - least_loaded: Assign to worker with least tasks
        - random: Random assignment

        Returns:
            Worker ID
        """
        if strategy == "least_loaded":
            return min(self.worker_loads.items(), key=lambda x: x[1])[0]
        elif strategy == "round_robin":
            return sum(self.worker_loads.values()) % self.num_workers
        else:  # random
            import random
            return random.randint(0, self.num_workers - 1)

    def update_load(self, worker_id: int, load_change: int):
        """Update worker load"""
        self.worker_loads[worker_id] += load_change


def parallel_test_prompts(prompts: List[str],
                          test_data: Any,
                          ground_truth: Any,
                          llm_service: Any,
                          metrics_evaluator: Any,
                          num_workers: int = None) -> List[Dict]:
    """
    Convenience function to test multiple prompts in parallel

    Args:
        prompts: List of prompt templates to test
        test_data: Test data
        ground_truth: Ground truth for evaluation
        llm_service: LLM service instance
        metrics_evaluator: Metrics evaluator instance
        num_workers: Number of parallel workers

    Returns:
        List of results for each prompt
    """
    # Create test configs
    test_configs = []
    for i, prompt in enumerate(prompts):
        test_configs.append({
            'prompt_id': f'prompt_{i}',
            'prompt': prompt,
            'test_data': test_data,
            'ground_truth': ground_truth,
            'llm_service': llm_service,
            'metrics_evaluator': metrics_evaluator
        })

    # Execute in parallel
    executor = DistributedTestExecutor(num_workers=num_workers)

    try:
        results = executor.execute_parallel_tests(
            test_configs,
            'agent.prompt_tuner.test_single_prompt'
        )
        return results
    finally:
        executor.shutdown()


class ProgressTracker:
    """
    Real-time progress tracker for distributed testing

    Provides callbacks and monitoring for long-running test suites.
    """

    def __init__(self):
        self.callbacks: List[Callable] = []
        self.history: List[Dict] = []

    def add_callback(self, callback: Callable):
        """Add progress callback"""
        self.callbacks.append(callback)

    def update(self, progress: Dict):
        """Update progress and trigger callbacks"""
        self.history.append({
            'timestamp': time.time(),
            **progress
        })

        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                print(f"Callback error: {e}")

    def print_progress(self, progress: Dict):
        """Simple progress printer"""
        print(
            f"\rProgress: {progress['completed']}/{progress['total_tasks']} "
            f"({progress['progress_pct']:.1f}%) - "
            f"Failed: {progress['failed']} - "
            f"Running: {progress['running']}",
            end='',
            flush=True
        )
