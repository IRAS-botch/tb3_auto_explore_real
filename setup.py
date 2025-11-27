from setuptools import setup
import os
from glob import glob

package_name = "tb3_auto_explore_real"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # [수정] launch 폴더의 모든 .launch.py 파일 포함
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # [수정] ★ 중요: param 폴더의 burger.yaml 설치 (이게 있어야 런치 파일이 읽음)
        ("share/" + package_name + "/param", glob("param/*.yaml")),
        # YOLO 스크립트와 모델 파일도 설치해 런치 파일에서 경로를 조회할 수 있도록 한다.
        ("share/" + package_name + "/yolo", glob("yolo/*")),
    ],
    install_requires=[
        "setuptools",
        "ultralytics>=8.2.0",
    ],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Frontier exploration node driving Nav2 to explore unknown space while mapping.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frontier_explorer = tb3_auto_explore_real.frontier_explorer:main",
        ],
    },
)
