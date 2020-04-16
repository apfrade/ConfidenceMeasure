import setuptools

with open("README.md", encoding="utf8") as readme_file:
    long_description = readme_file.read()

requirements = ["pandas>=0.2", "numpy>=1", "matplotlib>=3"]

setuptools.setup(
    name='confidence_tool',
    version="0.0.4",
    author='Andre Frade',
    py_modules=['confidence_tool.confidence_tool'],
    author_email="andre.frade@hertford.ox.ac.uk",
    description='Confidence tool package for classification models',
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