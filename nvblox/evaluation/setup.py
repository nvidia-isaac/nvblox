from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f.readlines() if line]

setup(
    name='nvblox_evaluation',
    version='0.0.0',
    description='Scripts for evaluating nvblox.',
    author='nvblox team.',
    author_email='amillane@nvidia.com + remos@nvidia.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages()
)
