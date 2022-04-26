===========
Integrators
===========

An integrator is, very generally, a class that modifies the content of a layers.

The integrators currently offered can be split into two types:

* Those which fuse incoming sensor data into a layer. For example, the :ref:`TsdfIntegrator <Class ProjectiveTsdfIntegrator>` and :ref:`ColorIntegrator <Class ProjectiveColorIntegrator>` fused depth images and color images into ``TsdfLayer`` and ``ColorLayer`` respectively.
* Those which transform the contents of one layer to update the data in another layer. For example the :ref:`EsdfIntegrator <Class EsdfIntegrator>` transforms an TSDF in a ``TsdfLayer`` into an ESDF in a ``EsdfLayer``.

The API for the currently available integrators are here:

* :ref:`TsdfIntegrator <Class ProjectiveTsdfIntegrator>`
* :ref:`ColorIntegrator <Class ProjectiveColorIntegrator>`
* :ref:`EsdfIntegrator <Class EsdfIntegrator>`
* :ref:`MeshIntegrator <Class MeshIntegrator>`

