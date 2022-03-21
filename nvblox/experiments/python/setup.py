import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f.readlines() if line]

setuptools.setup(
    name="nvblox_experiments",
    version="0.0.0",
    author="Helen Oleynikova + Alexander Millane",
    author_email="holeynikova@nvidia.com + amilllane@nvidia.com",
    description="nvblox_experiments: Helper module for nvblox experiments",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    include_package_data=True,
    packages=setuptools.find_packages()
)
