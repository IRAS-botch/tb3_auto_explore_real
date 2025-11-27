import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # === 1. Launch 인자 정의 ===
    # [수정] 실제 환경이므로 기본값을 false로 변경
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    slam = LaunchConfiguration("slam", default="true")
    use_rviz = LaunchConfiguration("use_rviz", default="false")

    # 내 패키지 경로
    tb3_auto_dir = get_package_share_directory("tb3_auto_explore_real")

    # 내 패키지의 burger.yaml 사용
    default_param_file = os.path.join(tb3_auto_dir, "param", "burger.yaml")

    params_file = LaunchConfiguration("params_file")

    # RViz 설정 파일 경로
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    rviz_config_file = os.path.join(nav2_bringup_dir, "rviz", "nav2_default_view.rviz")

    # === 2. 패키지 경로 및 설정 ===
    nav2_launch_file = os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
    slam_config_file = os.path.join(tb3_auto_dir, "param", "slam_toolbox.yaml")

    # === 3. Navigation2 실행 ===
    nav2_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": params_file,
            "autostart": "true",
        }.items(),
    )

    # === 4. SLAM Toolbox 실행 ===
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": slam_config_file,
        }.items(),
        condition=IfCondition(slam),
    )

    # === 5. Frontier Explorer 노드 ===
    explorer = Node(
        package="tb3_auto_explore_real",
        executable="frontier_explorer",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # === 6. RViz2 노드 ===
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # === 7. YOLO Image Topic 실행 (Remote PC에서 실행 가정) ===
    # 주의: 이 경로는 Remote PC에 실제로 존재해야 합니다.
    yolo_script_path = "/home/ubuntu/tb3_auto_explore_real/yolo/image_topic.py"

    yolo_node = ExecuteProcess(cmd=["python3", yolo_script_path], output="screen")

    # === 8. LaunchDescription 구성 ===
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="Use simulation clock"
        )
    )
    ld.add_action(
        DeclareLaunchArgument("slam", default_value="true", description="Run SLAM")
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_rviz", default_value="false", description="Start RViz2 automatically"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "params_file",
            default_value=default_param_file,
            description="Full path to the ROS2 parameters file to use for all launched nodes",
        )
    )

    ld.add_action(nav2_launch_include)
    ld.add_action(slam_toolbox_launch)
    ld.add_action(rviz_node)
    ld.add_action(explorer)
    ld.add_action(yolo_node)

    return ld
