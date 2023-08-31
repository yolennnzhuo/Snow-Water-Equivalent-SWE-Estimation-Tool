from setuptools import setup, find_packages

setup(
    name='swe_tool',
    version='0.1',
    packages=find_packages(),
    description="A tool for estimating Snow Water Equivalent (SWE)",
    author="Yulin Zhuo",
    author_email="yz6622@ic.ac.uk",
    url="https://github.com/ese-msc-2022/irp-yz6622.git",
    install_requires=[
        'numpy>=1.13.0',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'torch',
        'scikit-learn',
        'rasterio',
        'cartopy',
        'pytest',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
