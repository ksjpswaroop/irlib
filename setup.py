from setuptools import setup, find_packages
setup(
    name="irlib",
    version="0.3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11,<3.13",
)
