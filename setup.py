from setuptools import setup, find_packages

setup(
    name="slgp",  # Change this if you want a different install name
    version="0.1.0",
    author="Shujie Fan",
    description="Local Gaussian Process regression with differential evolution",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/VOD555/LGP_tools",  # Update if needed
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",  # Or whatever you use
    ],
    python_requires=">=3.8",
)