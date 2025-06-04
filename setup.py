from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any version specifiers and comments.
    """
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
    # Clean up the requirements
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
        
    return requirements
    
setup(
    name="ML_project",
    verison="0.0.1",
    author="Vedant",
    author_email="vedanttinkhede797@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)