#!/usr/bin/env python3
"""
Launch Gazebo Classic with the empty 8x8 meta-rl-tb3 validation world and 
spawn a FUNCTIONAL TurtleBot3 Burger (with sensor + diff-drive plugins) 
at the arena center, in one command.

"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _world_file() -> str:
    """Resolve the world file relative to THIS launch file (path-independent)."""
    here = os.path.dirname(os.path.abspath(__file__))
    world = os.path.normpath(os.path.join(here, "..", "worlds", "arena_8x8_empty.world"))
    if not os.path.isfile(world):
        raise FileNotFoundError(f"World file not found at {world}")
    return world


def _tb3_model_sdf() -> str:
    """Locate the TB3 Burger SDF model (this is the one WITH Gazebo plugins)."""
    tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    sdf = os.path.join(tb3_gazebo, "models", "turtlebot3_burger", "model.sdf")
    if not os.path.isfile(sdf):
        raise FileNotFoundError(
            f"TB3 burger SDF model not found at {sdf}. "
            f"Is ros-humble-turtlebot3-gazebo installed?"
        )
    return sdf


def generate_launch_description():
    world_file = _world_file()
    model_sdf = _tb3_model_sdf()
    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")
    tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")

    args = [
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("x", default_value="0.0",
                              description="Robot spawn X (arena center; bridge --spawn-xy must match)"),
        DeclareLaunchArgument("y", default_value="0.0",
                              description="Robot spawn Y"),
        DeclareLaunchArgument("yaw", default_value="0.0",
                              description="Robot spawn yaw (rad)"),
    ]

    # GAZEBO_MODEL_PATH so the TB3 model meshes and world model:// includes
    # (ground_plane, sun) resolve. Include the TB3 gazebo models dir.
    set_model_path = SetEnvironmentVariable(
        "GAZEBO_MODEL_PATH",
        [os.path.join(tb3_gazebo, "models"), ":", os.path.expanduser("~/.gazebo/models")],
    )

    # 1) Gazebo server + client via the gazebo_ros launch includes.
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_file, "verbose": "true"}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    # 2) Spawn the FUNCTIONAL robot from the SDF model (has the plugins), after
    #    Gazebo is genuinely up. Generous delay + -timeout for WSL2.
    spawn_robot = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_robot",
                output="screen",
                arguments=[
                    "-entity", "burger",
                    "-file", model_sdf,
                    "-x", LaunchConfiguration("x"),
                    "-y", LaunchConfiguration("y"),
                    "-z", "0.01",
                    "-Y", LaunchConfiguration("yaw"),
                    "-timeout", "60",
                ],
            )
        ],
    )

    return LaunchDescription([
        LogInfo(msg="=== Replicated 10x10 world + TB3 Burger (SDF, with plugins) ==="),
        LogInfo(msg=f"world: {world_file}"),
        LogInfo(msg=f"model: {model_sdf}"),
        *args,
        set_model_path,
        gzserver,
        gzclient,
        spawn_robot,
    ])



