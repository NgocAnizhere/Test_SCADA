"""Setup configuration cho Wind Turbine SCADA Fault Detection System."""

from setuptools import setup, find_packages
from pathlib import Path

# Đọc README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Đọc requirements
req_path = Path(__file__).parent / "requirements.txt"
requirements = []
if req_path.exists():
    requirements = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="wind-turbine-scada-fault-detection",
    version="1.0.0",
    author="SCADA Team",
    description="End-to-end Wind Turbine SCADA Fault Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "scada-detect=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
