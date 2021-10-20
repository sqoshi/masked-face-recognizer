from setuptools import setup, find_packages

setup(
    name="Face Recognizer",
    version="1.0.1",
    description="Tool to learn model recognizing masked person on image.",
    url="https://github.com/sqoshi/masked-face-recognizer",
    author="Piotr Popis",
    author_email="piotrpopis@icloud.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "coloredlogs~=15.0.1",
        "setuptools==44.0.0",
        "opencv-python==4.5.3.5",
        "imutils==0.5.4",
        "pandas>=1.3.2",
        "matplotlib==3.4.3",
        "scikit-learn==0.24.2"
    ],
)
