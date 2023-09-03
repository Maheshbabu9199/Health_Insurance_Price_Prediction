from setuptools import find_packages, setup
from typing import List

trigger="-e ."

def get_requirements(filepath)->List[str]:
    requirements = []
    with open(filepath, 'r') as f:
        requirements = [requirement.replace("\n","") for requirement in f.readlines()]
    if trigger in requirements:
        requirements.remove(trigger)

    return requirements


setup(name="Insurance_prediction", author="maheshbabu9199@gmail.com", author_email="maheshbabu", version="0.0.1", 
packages=find_packages(),install_requires=get_requirements("requirements.txt"))
