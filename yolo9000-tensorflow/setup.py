

from setuptools import setup, find_packages

setup(
    name='yolo9000-tf',
    version='0.1.0',
    description='Implementation of yolo v2',
    url='https://github.com/wen-kou/yolo.git',
    author='wen kou',
    author_email='wen.kou@shopee.com',
    packages=find_packages(exclude=['tests'])
)