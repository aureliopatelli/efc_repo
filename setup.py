import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="efc_repo",
    version="0.1",
    author="Aurelio Patelli",
    author_email="aurelio.patelli@cref.com",
    license='MIT',
    keywords="networks economics fitness complexity",
    description="Package for economic fitness and complexity framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aureliopatelli/efc_repo.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
                      "numpy>=1.14",
                      "scipy>=1.4",
                      "tqdm>=4.52.0",
                      "bicm>=3.3.0"
                      ],
    zip_safe=False,
)
