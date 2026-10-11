import os
from glob import glob

from setuptools import setup

data = [
    ("share/ament_index/resource_index/packages", ["resource/adaptmap_sim"]),
    ("share/adaptmap_sim", ["package.xml"]),
    ("share/adaptmap_sim/launch", glob("launch/*.py")),
    ("share/adaptmap_sim/worlds", glob("worlds/*")),
]
for d in glob("models/*"):
    data.append((os.path.join("share/adaptmap_sim", d), glob(d + "/*")))

setup(name="adaptmap_sim", version="0.1.0", packages=[], data_files=data, install_requires=["setuptools"])
