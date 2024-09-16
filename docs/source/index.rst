.. SDynPy documentation master file, created by
   sphinx-quickstart on Mon Jan 31 10:42:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: /images/logo_horizontal.svg
  :width: 550
  :alt: SDynPy Logo
  :align: center

|

Welcome to SDynPy's documentation!
==================================

|docs| |tests| |coverage| |lint| |version|

**SDynPy** is a package for performing structural dynamic analyses using Python.
It contains several objects that represent various structural dynamics data
types (shapes, data, geometry, etc.) as well as various functions and methods
to make working with these objects easy!

Check out the :doc:`usage` section for further information on how to use the package,
including how to :ref:`install <installation>` the package.

To quickly get running with SDynPy, check out the  out the :doc:`examples` for
an overview of recommended workflows and a summary of current functionality.

***********
Information
***********

- `Contributing <https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/blob/main/CONTRIBUTING.rst>`_
- `Documentation <http://structmechtools.cee-gitlab.lan/structural-dynamics-python-libraries/>`_
- `Project <https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/>`_
- `Releases <https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/releases>`_
- `Tutorial <http://structmechtools.cee-gitlab.lan/structural-dynamics-python-libraries/sdynpy_showcase.html>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
   usage
   sdynpy_showcase
   examples
   modal_tutorials
   modules
   bibliography

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..
    Badges ========================================================================

.. |docs| image:: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/badges/docs.svg?job=pages
    :target: http://structmechtools.cee-gitlab.lan/structural-dynamics-python-libraries/
    :alt: docs

.. |tests| image:: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/badges/tests.svg?job=basic-tests
    :target: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/logs/report.xml?job=basic-tests
    :alt: tests

.. |coverage| image:: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/badges/main/coverage.svg
    :target: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/pipelines/latest
    :alt: coverage

.. |lint| image:: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/badges/lint.svg?job=static-code-checks
    :target: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/logs/lint.log?job=static-code-checks
    :alt: lint

.. |version| image:: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/jobs/artifacts/main/raw/badges/version.svg?job=badges
    :target: https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries/-/releases
