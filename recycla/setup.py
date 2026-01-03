import setuptools

setuptools.setup(
    name="recycla",
    version="0.4.5",
    author="Chris Minar",
    author_email="chris.minar@gmail.com",
    description="A package for classifying recyclable materials.",
    url="",
    packages=setuptools.find_packages("src", exclude=["tests"]),
    package_dir={"": "src"},
    package_data={
        "recycla.config": ["*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "numpy==1.26.4",
        "torchvision",
        "Pillow",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "recycla = recycla.entrypoint:recycla",
        ],
    },
)
