import torch
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="mae-lite",
    version="0.0.1",
    author="Wangsr",
    author_email="wangsr126@gmail.com",
    description="Self Supervised Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "ssl_train=mae_lite.tools.train:main",
            "ssl_eval=mae_lite.tools.eval:main",
        ]
    },
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
