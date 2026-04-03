# setup.py — CropAI Package Setup

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cropai",
    version="1.0.0",
    author="CropAI Team",
    author_email="cropai@project.edu",
    description="AI-powered Crop Disease Detection & Yield Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cropai/cropai",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Education",
        "Development Status :: 4 - Beta",
    ],
    entry_points={
        "console_scripts": [
            "cropai-train=train:main",
            "cropai-api=api:app",
        ],
    },
    keywords="crop disease detection yield prediction AI ML deep-learning",
    project_urls={
        "Documentation": "https://github.com/cropai/cropai/wiki",
        "Bug Tracker":   "https://github.com/cropai/cropai/issues",
    },
)
