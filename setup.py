import io
import os
from pathlib import Path

from setuptools import find_packages, setup


NAME = 'pred_model'
DESCRIPTION = 'Mushroom Classification Model'
URL = 'https://github.com/akashsp7'
EMAIL = 'akashpawar9619@gmail.com'
AUTHOR = 'Akash Pawar'
REQUIRES_PYTHON = '>=3.12.0'

pwd = os.path.abspath(os.path.dirname(__file__))

def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
# about = {}
# with open(PACKAGE_DIR / 'VERSION') as f:
#     _version = f.read().strip()
#     about['__version__'] = _version

setup(
    name=NAME,
    version='1.0.0',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
)