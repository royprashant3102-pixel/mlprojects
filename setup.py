from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            req = line.strip()
            if not req or req.startswith('#') or '-e' in req:
                continue
            requirements.append(req)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Prashant',
    author_email='bhaskarshamoray11@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
