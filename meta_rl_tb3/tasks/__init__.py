"""Task distribution system for meta-learning."""

from meta_rl_tb3.tasks.task import Task, TaskConfig, TaskType
from meta_rl_tb3.tasks.distributions import (
    TaskDistribution,
    GoalReachingDistribution,
    GoalReachingDistributionConfig,
    ObstacleAvoidanceDistribution,
    ObstacleAvoidanceDistributionConfig,
    PathFollowingDistribution,
    PathFollowingDistributionConfig,
    MixedTaskDistribution,
    create_default_task_distribution,
    DynamicsDistribution,
    get_fixed_test_tasks,
)

__all__ = [
    "Task",
    "TaskConfig",
    "TaskType",
    "TaskDistribution",
    "GoalReachingDistribution",
    "GoalReachingDistributionConfig",
    "ObstacleAvoidanceDistribution",
    "ObstacleAvoidanceDistributionConfig",
    "PathFollowingDistribution",
    "PathFollowingDistributionConfig",
    "MixedTaskDistribution",
    "create_default_task_distribution",
    "DynamicsDistribution",
    "get_fixed_test_tasks",
]
