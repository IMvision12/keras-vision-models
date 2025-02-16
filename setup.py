from setuptools import find_packages, setup

setup(
    name="keras-vision-models",
    version="0.1",
    description="A library that offers Keras3 models, including popular computer vision architectures like ResNet, EfficientNet, and MobileNet, with pretrained weights, supporting transfer learning, object detection, image segmentation, and other functionalities.",
    url="https://github.com/IMvision12/keras-vision-models",
    author="Gitesh Chawda",
    author_email="gitesh.ch.0912@gmail.com",
    license="Apache License 2.0",
    packages=find_packages(),
    readme="README.md",
    install_requires=["keras", "tensorflow", "torch", "jax", "numpy", "timm"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
