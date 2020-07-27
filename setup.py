import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="softml-Paroag", # Replace with your own username
    version="0.0.1",
    author="Oscar Barbier",
    author_email="o.barbierd@gmail.com",
    description="package for soft testing ML algorithms on regression problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Paroag/softTestMachineLearning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)