from glob import glob

from setuptools import setup

setup(
    name="adaptmap_nav",
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/adaptmap_nav"]),
        ("share/adaptmap_nav", ["package.xml"]),
        ("share/adaptmap_nav/launch", glob("launch/*.py")),
        ("share/adaptmap_nav/params", glob("params/*")),
    ],
    install_requires=["setuptools"],
)
