import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LDRM",
    version="0.0.1",
    author="Anon",
    description="Package for using LDRMs for PEFT",
    long_description="TODO",
    long_description_content_type="text/markdown",
    packages=['src'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)