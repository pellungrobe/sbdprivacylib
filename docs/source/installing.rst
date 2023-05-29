**********************
Installing sbdprivlib
***********************

Before installing ``sbdprivlib``, you need to have setuptools installed.

=============
Quick install
=============

Get ``sbdprivlib`` from the Python Package Index at pypl_.

or install it with

.. code-block:: python

    pip install sbdprivlib

and an attempt will be made to find and install an appropriate version that matches your operating system and Python version.
Please note that ``sbdprivlib`` requires Python>=3.10


======================
Installing from source
======================

You can install from source by downloading a source archive file (tar.gz or zip) or by checking out the source files from the GitHub source code repository.

``sbdprivlib`` is a pure Python package; you don’t need a compiler to build or install it.

-------------------
Source archive file
-------------------
Download the source (tar.gz or zip file) from pypl_  or get the latest development version from GitHub_

Unpack and change directory to the source directory (it should have the files README.txt and setup.py).

Run python setup.py install to build and install

------
GitHub
------
Clone the ``sbdprivlib`` repostitory (see GitHub_ for options)

.. code-block:: python

    git clone https://github.com/pellungrobe/sbdprivacylib.git

Change directory to project

Run python setup.py install to build and install

If you don’t have permission to install software on your system, you can install into another directory using the --user, --prefix, or --home flags to setup.py.

For example

.. code-block:: python

    python setup.py install --prefix=/home/username/python

or

.. code-block:: python

    python setup.py install --home=~

or

.. code-block:: python

    python setup.py install --user

If you didn’t install in the standard Python site-packages directory you will need to set your PYTHONPATH variable to the alternate location. See http://docs.python.org/2/install/index.html#search-path for further details.

============
Requirements
============

python>=3.10

numpy>=1.24.3

pandas>=1.5.3

ipython>=8.12.0

tqdm>=4.65.0

nltk>=3.7

setuptools>=66.0.0

pyroaring>=0.4.2

pyfim>=6.28

To use ``sbdprivlib`` you need Python 3.10 or later.

The easiest way to get Python and most optional packages is to install the Enthought Python distribution “Canopy” or using Anaconda.

There are several other distributions that contain the key packages you need for scientific computing.


.. _pypl: https://pypi.python.org/pypi/sbdprivlib/
.. _GitHub: https://github.com/pellungrobe/sbdprivlib/