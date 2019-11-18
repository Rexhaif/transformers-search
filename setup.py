import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformers-search",
    version="0.0.1",
    author="Daniil Larionov",
    author_email="rexhaif.io@gmail.com",
    description="Tool for automatic nlp dataset creation through similarity search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rexhaif/transformers-search",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)