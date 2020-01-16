import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

requirements = ["pandas>=0.2", "numpy>=1"]

setuptools.setup(
    name='ConfidenceMeasure',
    version="0.0.3",
    author='Andre Frade',
    py_modules=['ConfidenceMeasure'],
    author_email="andre.frade@hertford.ox.ac.uk",
    description='Confidence system package for classification models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apfrade/ConfidenceMeasure.git",
    packages=setuptools.find_packages(),
    install_requires= requirements,
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
],  
)