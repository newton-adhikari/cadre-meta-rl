"""Task distribution classes for meta-learning.

This module provides task distributions for sampling tasks during
meta-training and evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

import numpy as np

from meta_rl_tb3.tasks.task import Task, TaskConfig, TaskType, estimate_task_difficulty


class TaskDistribution(ABC):
    """Abstract base class for task distributions.
    
    A task distribution defines how to sample tasks for meta-learning.
    Each distribution can generate tasks with varying parameters
    within specified bounds.
    """
    
    def __init__(self, seed: Optional[int] = None):
        # seed: Random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        self._seed = seed
    
    def seed(self, seed: int) -> None:
        self._seed = seed
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def sample(self) -> Task:
        """Sample a single task from the distribution.
        
        """
        pass
    
    def sample_batch(self, n: int) -> List[Task]:
        """Sample a batch of tasks.
        
        """
        return [self.sample() for _ in range(n)]
    
    def get_train_test_split(
        self,
        n_train: int,
        n_test: int,
    ) -> Tuple[List[Task], List[Task]]:
        """Generate train and test task splits.
        
        Ensures test tasks are distinct from training tasks.
        
        """
        # Sample more tasks and split
        all_tasks = self.sample_batch(n_train + n_test)
        train_tasks = all_tasks[:n_train]
        test_tasks = all_tasks[n_train:]
        return train_tasks, test_tasks


@dataclass
class GoalReachingDistributionConfig:
    """Configuration for goal-reaching task distribution.
    
    Attributes:
        arena_size: Arena size
        goal_distance_range: (min, max) distance of goal from start
        start_position_range: Bounds for start position sampling
        randomize_start: Whether to randomize start position
        max_episode_steps: Maximum episode steps (should match MAML config)
    """
    arena_size: float = 4.0
    goal_distance_range: Tuple[float, float] = (1.0, 3.0)
    start_position_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    randomize_start: bool = True
    max_episode_steps: int = 200  # Changed from 500 to match typical MAML training


class GoalReachingDistribution(TaskDistribution):
    """Distribution over goal-reaching tasks.
    
    Samples tasks with varying goal positions and optionally
    varying start positions.
    """
    
    def __init__(
        self,
        config: Optional[GoalReachingDistributionConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize goal-reaching distribution.

        """
        super().__init__(seed)
        self.config = config or GoalReachingDistributionConfig()
    
    def sample(self) -> Task:
        """Sample a goal-reaching task."""
        arena = self.config.arena_size
        margin = 0.5
        
        # Sample start position
        if self.config.randomize_start:
            if self.config.start_position_range:
                x_range, y_range = self.config.start_position_range
                start_x = self.rng.uniform(x_range[0], x_range[1])
                start_y = self.rng.uniform(y_range[0], y_range[1])
            else:
                start_x = self.rng.uniform(-arena + margin, arena - margin)
                start_y = self.rng.uniform(-arena + margin, arena - margin)
            start_position = (start_x, start_y)
            start_orientation = self.rng.uniform(-np.pi, np.pi)
        else:
            start_position = (0.0, 0.0)
            start_orientation = 0.0
        
        # Sample goal position at appropriate distance
        min_dist, max_dist = self.config.goal_distance_range
        
        for _ in range(100):  # Max attempts
            goal_angle = self.rng.uniform(-np.pi, np.pi)
            goal_distance = self.rng.uniform(min_dist, max_dist)
            
            goal_x = start_position[0] + goal_distance * np.cos(goal_angle)
            goal_y = start_position[1] + goal_distance * np.sin(goal_angle)
            
            # Check bounds
            if -arena + margin < goal_x < arena - margin and \
               -arena + margin < goal_y < arena - margin:
                break
        
        goal_position = (goal_x, goal_y)
        
        # Create task config
        task_config = TaskConfig(
            task_type=TaskType.GOAL_REACHING,
            arena_size=arena,
            max_episode_steps=self.config.max_episode_steps,
            goal_position=goal_position,
            start_position=start_position,
            start_orientation=start_orientation,
        )
        
        # Estimate difficulty
        task_config.difficulty = estimate_task_difficulty(task_config)
        
        return Task(config=task_config)


@dataclass
class ObstacleAvoidanceDistributionConfig:
    """Configuration for obstacle avoidance task distribution.
    
    Attributes:
        arena_size: Arena size
        num_obstacles_range: (min, max) number of obstacles
        obstacle_size_range: (min, max) obstacle size
        include_goal: Whether to include goal-reaching
        goal_distance_range: Distance range for goal (if included)
        max_episode_steps: Maximum episode steps (should match MAML config)
    """
    arena_size: float = 4.0
    num_obstacles_range: Tuple[int, int] = (3, 8)
    obstacle_size_range: Tuple[float, float] = (0.2, 0.5)
    include_goal: bool = True
    goal_distance_range: Tuple[float, float] = (1.5, 3.5)
    max_episode_steps: int = 200  # Changed from 500 to match typical MAML training


class ObstacleAvoidanceDistribution(TaskDistribution):
    """Distribution over obstacle avoidance tasks.
    
    Samples tasks with varying numbers and positions of obstacles.
    """
    
    def __init__(
        self,
        config: Optional[ObstacleAvoidanceDistributionConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize obstacle avoidance distribution."""
        super().__init__(seed)
        self.config = config or ObstacleAvoidanceDistributionConfig()
    
    def sample(self) -> Task:
        """Sample an obstacle avoidance task."""
        arena = self.config.arena_size
        margin = 0.5
        
        # Sample start position
        start_x = self.rng.uniform(-arena + margin, arena - margin)
        start_y = self.rng.uniform(-arena + margin, arena - margin)
        start_position = (start_x, start_y)
        start_orientation = self.rng.uniform(-np.pi, np.pi)
        
        # Sample goal if included
        goal_position = None
        if self.config.include_goal:
            min_dist, max_dist = self.config.goal_distance_range
            for _ in range(100):
                goal_angle = self.rng.uniform(-np.pi, np.pi)
                goal_distance = self.rng.uniform(min_dist, max_dist)
                
                goal_x = start_x + goal_distance * np.cos(goal_angle)
                goal_y = start_y + goal_distance * np.sin(goal_angle)
                
                if -arena + margin < goal_x < arena - margin and \
                   -arena + margin < goal_y < arena - margin:
                    goal_position = (goal_x, goal_y)
                    break
        
        # Sample obstacles
        min_obs, max_obs = self.config.num_obstacles_range
        num_obstacles = self.rng.randint(min_obs, max_obs + 1)
        
        obstacles = []
        for i in range(num_obstacles):
            for _ in range(50):  # Max attempts per obstacle
                obs_x = self.rng.uniform(-arena + margin, arena - margin)
                obs_y = self.rng.uniform(-arena + margin, arena - margin)
                
                # Check distance from start
                dist_to_start = np.sqrt((obs_x - start_x)**2 + (obs_y - start_y)**2)
                
                # Check distance from goal
                if goal_position:
                    dist_to_goal = np.sqrt((obs_x - goal_position[0])**2 + (obs_y - goal_position[1])**2)
                else:
                    dist_to_goal = float('inf')
                
                # Check distance from other obstacles
                min_obs_dist = float('inf')
                for obs in obstacles:
                    d = np.sqrt((obs_x - obs["position"][0])**2 + (obs_y - obs["position"][1])**2)
                    min_obs_dist = min(min_obs_dist, d)
                
                if dist_to_start > 1.0 and dist_to_goal > 0.8 and min_obs_dist > 0.6:
                    size_min, size_max = self.config.obstacle_size_range
                    size = self.rng.uniform(size_min, size_max)
                    
                    obstacles.append({
                        "position": (obs_x, obs_y),
                        "size": (size, size, 0.5),
                        "type": "box" if self.rng.random() > 0.5 else "cylinder",
                    })
                    break
        
        # Create task config
        task_config = TaskConfig(
            task_type=TaskType.OBSTACLE_AVOIDANCE,
            arena_size=arena,
            max_episode_steps=self.config.max_episode_steps,
            goal_position=goal_position,
            start_position=start_position,
            start_orientation=start_orientation,
            obstacles=obstacles,
        )
        
        task_config.difficulty = estimate_task_difficulty(task_config)
        
        return Task(config=task_config)


@dataclass
class PathFollowingDistributionConfig:
    """Configuration for path following task distribution.
    
    Attributes:
        arena_size: Arena size
        num_waypoints_range: (min, max) number of waypoints
        path_types: List of allowed path types
        max_episode_steps: Maximum episode steps (should match MAML config)
    """
    arena_size: float = 4.0
    num_waypoints_range: Tuple[int, int] = (5, 15)
    path_types: List[str] = None
    max_episode_steps: int = 200  # Changed from 500 to match typical MAML training
    
    def __post_init__(self):
        if self.path_types is None:
            self.path_types = ["straight", "circular", "sine_wave", "random"]


class PathFollowingDistribution(TaskDistribution):
    """Distribution over path following tasks.
    
    Samples tasks with varying path shapes and complexities.
    """
    
    def __init__(
        self,
        config: Optional[PathFollowingDistributionConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize path following distribution."""
        super().__init__(seed)
        self.config = config or PathFollowingDistributionConfig()
    
    def _generate_straight_path(self, num_points: int) -> List[Tuple[float, float]]:
        """Generate a straight line path."""
        arena = self.config.arena_size
        margin = 0.5
        
        start = (
            self.rng.uniform(-arena + margin, arena - margin),
            self.rng.uniform(-arena + margin, arena - margin),
        )
        angle = self.rng.uniform(-np.pi, np.pi)
        
        waypoints = []
        for i in range(num_points):
            t = i * 0.5
            x = start[0] + t * np.cos(angle)
            y = start[1] + t * np.sin(angle)
            
            # Clip to arena
            x = np.clip(x, -arena + margin, arena - margin)
            y = np.clip(y, -arena + margin, arena - margin)
            waypoints.append((x, y))
        
        return waypoints
    
    def _generate_circular_path(self, num_points: int) -> List[Tuple[float, float]]:
        """Generate a circular path."""
        arena = self.config.arena_size
        radius = self.rng.uniform(1.0, min(2.0, arena - 0.5))
        
        center_x = self.rng.uniform(-arena + radius + 0.5, arena - radius - 0.5)
        center_y = self.rng.uniform(-arena + radius + 0.5, arena - radius - 0.5)
        
        waypoints = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoints.append((x, y))
        
        return waypoints
    
    def _generate_sine_path(self, num_points: int) -> List[Tuple[float, float]]:
        """Generate a sinusoidal path."""
        arena = self.config.arena_size
        margin = 0.5
        amplitude = self.rng.uniform(0.5, 1.5)
        
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = -arena + margin + (2 * arena - 2 * margin) * t
            y = amplitude * np.sin(2 * np.pi * t)
            waypoints.append((x, y))
        
        return waypoints
    
    def _generate_random_path(self, num_points: int) -> List[Tuple[float, float]]:
        """Generate a random waypoint path."""
        arena = self.config.arena_size
        margin = 0.5
        
        waypoints = [(
            self.rng.uniform(-arena + margin, arena - margin),
            self.rng.uniform(-arena + margin, arena - margin),
        )]
        
        for _ in range(num_points - 1):
            prev = waypoints[-1]
            
            for _ in range(50):
                angle = self.rng.uniform(-np.pi, np.pi)
                dist = self.rng.uniform(0.5, 1.5)
                
                x = prev[0] + dist * np.cos(angle)
                y = prev[1] + dist * np.sin(angle)
                
                if -arena + margin < x < arena - margin and \
                   -arena + margin < y < arena - margin:
                    waypoints.append((x, y))
                    break
            else:
                # Fallback
                waypoints.append((
                    self.rng.uniform(-arena + margin, arena - margin),
                    self.rng.uniform(-arena + margin, arena - margin),
                ))
        
        return waypoints
    
    def sample(self) -> Task:
        """Sample a path following task."""
        # Sample path type
        path_type = self.rng.choice(self.config.path_types)
        
        # Sample number of waypoints
        min_wp, max_wp = self.config.num_waypoints_range
        num_waypoints = self.rng.randint(min_wp, max_wp + 1)
        
        # Generate path
        if path_type == "straight":
            waypoints = self._generate_straight_path(num_waypoints)
        elif path_type == "circular":
            waypoints = self._generate_circular_path(num_waypoints)
        elif path_type == "sine_wave":
            waypoints = self._generate_sine_path(num_waypoints)
        else:
            waypoints = self._generate_random_path(num_waypoints)
        
        # Start position is first waypoint
        start_position = waypoints[0]
        
        # Start orientation toward second waypoint
        if len(waypoints) > 1:
            dx = waypoints[1][0] - waypoints[0][0]
            dy = waypoints[1][1] - waypoints[0][1]
            start_orientation = np.arctan2(dy, dx)
        else:
            start_orientation = 0.0
        
        # Create task config
        task_config = TaskConfig(
            task_type=TaskType.PATH_FOLLOWING,
            arena_size=self.config.arena_size,
            max_episode_steps=self.config.max_episode_steps,
            start_position=start_position,
            start_orientation=start_orientation,
            path_waypoints=waypoints,
            path_type=path_type,
        )
        
        task_config.difficulty = estimate_task_difficulty(task_config)
        
        return Task(config=task_config)


class MixedTaskDistribution(TaskDistribution):
    """Distribution that mixes multiple task types.
    
    Samples tasks from different distributions according to
    specified weights.

    """
    
    def __init__(
        self,
        distributions: List[Tuple[TaskDistribution, float]],
        seed: Optional[int] = None,
    ):
        """Initialize mixed distribution.

        """
        super().__init__(seed)
        
        self.distributions = [d for d, _ in distributions]
        weights = np.array([w for _, w in distributions])
        self.weights = weights / weights.sum()  # Normalize
    
    def seed(self, seed: int) -> None:
        """Set random seed for all distributions."""
        super().seed(seed)
        for dist in self.distributions:
            dist.seed(seed)
    
    def sample(self) -> Task:
        """Sample a task from one of the distributions."""
        idx = self.rng.choice(len(self.distributions), p=self.weights)
        return self.distributions[idx].sample()
    
    def sample_balanced_batch(self, n_per_type: int) -> List[Task]:
        """Sample equal numbers from each distribution.

        """
        tasks = []
        for dist in self.distributions:
            tasks.extend(dist.sample_batch(n_per_type))
        
        # Shuffle
        self.rng.shuffle(tasks)
        return tasks


def create_default_task_distribution(
    task_types: Optional[List[TaskType]] = None,
    seed: Optional[int] = None,
) -> TaskDistribution:
    """Create a default mixed task distribution.

    """
    if task_types is None:
        task_types = [TaskType.GOAL_REACHING, TaskType.OBSTACLE_AVOIDANCE, TaskType.PATH_FOLLOWING]
    
    distributions = []
    
    if TaskType.GOAL_REACHING in task_types:
        distributions.append((GoalReachingDistribution(seed=seed), 1.0))
    
    if TaskType.OBSTACLE_AVOIDANCE in task_types:
        distributions.append((ObstacleAvoidanceDistribution(seed=seed), 1.0))
    
    if TaskType.PATH_FOLLOWING in task_types:
        distributions.append((PathFollowingDistribution(seed=seed), 1.0))
    
    if len(distributions) == 1:
        return distributions[0][0]
    
    return MixedTaskDistribution(distributions, seed=seed)


# =============================================================================
# DynamicsDistribution — samples DynamicsConfig for per-task deviations
# =============================================================================

class DynamicsDistribution:
    """Samples DynamicsConfig objects from a named split of DYNAMICS_RANGES.

    """

    def __init__(self, split: str = "train", seed: Optional[int] = None):
        pass

