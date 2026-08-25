from glob import glob

from setuptools import find_packages, setup

package_name = "liquid_depth_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Liquid Depth Project",
    maintainer_email="noreply@example.com",
    description="Synchronized RGB-D capture for the liquid depth pipeline",
    license="MIT",
    entry_points={"console_scripts": ["capture_node = liquid_depth_camera.capture_node:main"]},
)

