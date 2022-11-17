# Integrators

An integrator is, very generally, a class that modifies the content of a layers.

The integrators currently offered can be split into two types:

* Those which fuse incoming sensor data into a layer. For example, the [TsdfIntegrator](@ref nvblox::ProjectiveTsdfIntegrator) and [ColorIntegrator](@ref nvblox::ProjectiveColorIntegrator) fused depth images and color images into [TsdfLayer](@ref nvblox::TsdfLayer) and [ColorLayer](@ref nvblox::ColorLayer) respectively.
* Those which transform the contents of one layer to update the data in another layer. For example the [EsdfIntegrator](@ref nvblox::EsdfIntegrator) transforms an TSDF in a [TsdfLayer](@ref nvblox::TsdfLayer) into an ESDF in a [EsdfLayer](@ref nvblox::EsdfLayer).

## API

The API for the currently available integrators are here:

* [TsdfIntegrator](@ref nvblox::ProjectiveTsdfIntegrator)
* [ColorIntegrator](@ref nvblox::ProjectiveColorIntegrator)
* [EsdfIntegrator](@ref nvblox::EsdfIntegrator)
* [MeshIntegrator](@ref nvblox::MeshIntegrator)

