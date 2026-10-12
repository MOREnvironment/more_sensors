from setuptools import find_packages, setup


setup(
    name="more_sensors",
    version="0.1.0",
    packages=find_packages(include=["more_sensors", "more_sensors.*"]),
    include_package_data=True,
    install_requires=[
        "casadi>=3.5.5",
        "more_common>=0.1.0",
        "numpy",
    ],
)
