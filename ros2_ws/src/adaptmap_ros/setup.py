from glob import glob

from setuptools import setup

setup(
    name="adaptmap_ros",
    version="0.1.0",
    packages=["adaptmap_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/adaptmap_ros"]),
        ("share/adaptmap_ros", ["package.xml"]),
        ("share/adaptmap_ros/launch", glob("launch/*.py")),
        ("share/adaptmap_ros/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "adaptmap_node = adaptmap_ros.adaptmap_node:main",
        "waypoint_driver = adaptmap_ros.waypoint_driver:main",
    ]},
)
