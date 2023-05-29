from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = "Pellungrini"
__license__ = "BSD-2-Clause"
__email__ = "roberto.pellungrini@sns.it"


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="privlib",
    version="0.0.1",
    license="BSD-Clause-2",
    description="Privacy library of the SoBigData research infrastructure",
    url="https://github.com/USERNAME/project",
    author="Roberto Pellungrini",
    author_email="roberto.pellungrini@sns.it",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Other",
        "Operating System :: MacOS",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    keywords="keyword1 keyword2 keyword3",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={"flag": []},
    packages=find_packages(
        exclude=[
            "*.test",
            "*.tests",
            "tests.*",
            "*.test.*",
            "*.tests.*",
            "test.*",
            "test",
            "privlib.test",
            "privlib.test.*",
        ]
    ),
)
