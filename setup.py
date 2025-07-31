from setuptools import setup, find_packages

setup(
    name="qiskit_intro",
    version="0.1",
    description="Quantum circuit optimisation using reinforcement learning and simulation benchmarks",
    author="miles5551",
    author_email="miles5551.mm@gmail.com",
    packages=find_packages(),  # automatically finds all packages with __init__.py
    install_requires=[
        "qiskit",
        "numpy",
        "tqdm",
        "matplotlib",
        "scipy",
        
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
        "Operating System :: OS Independent",
    ],
)
