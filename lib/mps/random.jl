@cenum MPSMatrixRandomDistribution::NSUInteger begin
    # Generate random bits according to the distribution of the underlying generator.
    MPSMatrixRandomDistributionDefault = 0
    # Generate uniformly distributed random floating point values in the interval [0, 1).
    MPSMatrixRandomDistributionUniform = 1 << 0
    # Generate normally distributed random floating point values.
    MPSMatrixRandomDistributionNormal = 1 << 1
end

export MPSMatrixDescriptor

@objcwrapper immutable=false MPSMatrixRandomDistributionDescriptor <: NSObject

@objcproperties MPSMatrixRandomDistributionDescriptor begin
    @autoproperty distributionType::MPSMatrixRandomDistribution setter=setDistributionType
    @autoproperty minimum::Cfloat setter=setMinimum
    @autoproperty maximum::Cfloat setter=setMaximum
    @autoproperty mean::Cfloat setter=setMean
    @autoproperty standardDeviation::Cfloat setter=setStandardDeviation
end


function MPSMatrixDefaultDistributionDescriptor()
    desc = @objc [MPSMatrixRandomDistributionDescriptor defaultDistributionDescriptor]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixUniformDistributionDescriptor(minimum, maximum)
    desc = @objc [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:minimum::Cfloat
                                      maximum:maximum::Cfloat]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixNormalDistributionDescriptor(mean, standardDeviation)
    desc = @objc [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:mean::Cfloat
                                      standardDeviation:standardDeviation::Cfloat]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixNormalDistributionDescriptor(mean, standardDeviation, minimum, maximum)
    desc = @objc [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:mean::Cfloat
                                      standardDeviation:standardDeviation::Cfloat
                                      minimum:minimum::Cfloat
                                      maximum:maximum::Cfloat]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end



export MPSMatrixRandom

@objcwrapper immutable=false MPSMatrixRandom <: MPSKernel

@objcproperties MPSMatrixRandom begin
    @autoproperty destinationDataType::MPSDataType
    @autoproperty distributionType::MPSMatrixRandomDistribution
    @autoproperty batchStart::NSUInteger setter=setBatchStart
    @autoproperty batchSize::NSUInteger setter=setBatchSize
end


function MPSMatrixRandom(device)
    kernel = @objc [MPSMatrixRandom alloc]::id{MPSMatrixRandom}
    obj = MPSMatrixRandom(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandom} initWithDevice:device::id{MTLDevice}]::id{MPSMatrixRandom}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixRandom, destinationVector::MPSVector)
    @objc [kernel::id{MPSMatrixDecompositionLU} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                destinationVector:destinationVector::id{MPSVector}]::Nothing
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixRandom, destinationMatrix::MPSMatrix)
    @objc [kernel::id{MPSMatrixDecompositionLU} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                destinationMatrix:destinationMatrix::id{MPSMatrix}]::Nothing
end


export MPSMatrixRandomMTGP32

@objcwrapper immutable=false MPSMatrixRandomMTGP32 <: MPSMatrixRandom


"""
Generates random numbers using a Mersenne Twister algorithm suitable for GPU execution.
It uses a period of 2**11214. 
For further details see: Mutsuo Saito. A Variant of Mersenne Twister Suitable for Graphic Processors. arXiv:1005.4973
"""
function MPSMatrixRandomMTGP32(device)
    kernel = @objc [MPSMatrixRandomMTGP32 alloc]::id{MPSMatrixRandomMTGP32}
    obj = MPSMatrixRandomMTGP32(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomMTGP32} initWithDevice:device::id{MTLDevice}]::id{MPSMatrixRandomMTGP32}
    return obj
end

function MPSMatrixRandomMTGP32(device, destinationDataType, seed)
    kernel = @objc [MPSMatrixRandomMTGP32 alloc]::id{MPSMatrixRandomMTGP32}
    obj = MPSMatrixRandomMTGP32(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomMTGP32} initWithDevice:device::id{MTLDevice}
                                          destinationDataType:destinationDataType::MPSDataType
                                          seed:seed::NSUInteger]::id{MPSMatrixRandomMTGP32}
    return obj
end

function MPSMatrixRandomMTGP32(device, destinationDataType, seed, distributionDescriptor)
    kernel = @objc [MPSMatrixRandomMTGP32 alloc]::id{MPSMatrixRandomMTGP32}
    obj = MPSMatrixRandomMTGP32(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomMTGP32} initWithDevice:device::id{MTLDevice}
                                          destinationDataType:destinationDataType::MPSDataType
                                          seed:seed::NSUInteger
                                          distributionDescriptor:distributionDescriptor::id{MPSMatrixRandomDistributionDescriptor}]::id{MPSMatrixRandomMTGP32}
    return obj
end

function synchronizeState(cmdbuf)
    @objc [kernel::id{MPSMatrixRandomMTGP32} synchronizeStateOnCommandBuffer:cmdbuf::id{MTLCommandBuffer}]::Nothing
end


export MPSMatrixRandomPhilox


@objcwrapper immutable=false MPSMatrixRandomPhilox <: MPSMatrixRandom

"""
Generates random numbers using a counter based algorithm. 
For further details see:
John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel Random Numbers: As Easy as 1, 2, 3.
"""
function MPSMatrixRandomPhilox(device)
    kernel = @objc [MPSMatrixRandomPhilox alloc]::id{MPSMatrixRandomPhilox}
    obj = MPSMatrixRandomPhilox(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomPhilox} initWithDevice:device::id{MTLDevice}]::id{MPSMatrixRandomPhilox}
    return obj
end

function MPSMatrixRandomPhilox(device, destinationDataType, seed)
    kernel = @objc [MPSMatrixRandomPhilox alloc]::id{MPSMatrixRandomPhilox}
    obj = MPSMatrixRandomPhilox(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomMTGP32} initWithDevice:device::id{MTLDevice}
                                          destinationDataType:destinationDataType::MPSDataType
                                          seed:seed::NSUInteger]::id{MPSMatrixRandomPhilox}
    return obj
end

function MPSMatrixRandomPhilox(device, destinationDataType, seed, distributionDescriptor)
    kernel = @objc [MPSMatrixRandomPhilox alloc]::id{MPSMatrixRandomPhilox}
    obj = MPSMatrixRandomPhilox(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomPhilox} initWithDevice:device::id{MTLDevice}
                                          destinationDataType:destinationDataType::MPSDataType
                                          seed:seed::NSUInteger
                                          distributionDescriptor:distributionDescriptor::id{MPSMatrixRandomDistributionDescriptor}]::id{MPSMatrixRandomPhilox}
    return obj
end



######



# interfacing with Random standard library

using Random


make_seed() = Base.rand(RandomDevice(), UInt64)

mutable struct MPSPhiloxRNG <: Random.AbstractRNG
    seed::NSUInteger
    device::MTLDevice

    function MPSPhiloxRNG(seed=make_seed(), device=current_device())
        new(seed, device)
    end
end

export MPSPhiloxRNG

## seeding

function Random.seed!(rng::MPSPhiloxRNG, seed=make_seed())
    rng.seed = seed
end

Random.seed!(rng::MPSPhiloxRNG, ::Nothing) = Random.seed!(rng)


## in-place

# uniform
const UniformType = Union{Type{Float32},Type{UInt32}}
const UniformArray = MtlArray{<:Union{Float32,UInt32}}

function Random.rand!(rng::MPSPhiloxRNG, A::MtlArray{Float32})
    dtype = jl_typ_to_mps[Float32]
    dist = MPSMatrixUniformDistributionDescriptor(0.0, 1.0)
    kernel = MPSMatrixRandomPhilox(rng.device, dtype, rng.seed, dist)
    cmdbuf = MTLCommandBuffer(global_queue(rng.device))
    encode!(cmdbuf, kernel, MPSVector(A))
    commit!(cmdbuf)
    rng.seed = make_seed()
    return A
end

# 
# # normal# const NormalType = Union{Type{Float32},Type{Float64}}
# const NormalArray = DenseCuArray{<:Union{Float32,Float64}}
# function Random.randn!(rng::RNG, A::DenseCuArray{Float32}; mean=0, stddev=1)
#     update_stream(rng)
#     inplace_pow2(A, B -> curandGenerateNormal(rng, B, length(B), mean, stddev))
#     return A
# end
# function Random.randn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
#     update_stream(rng)
#     inplace_pow2(A, B -> curandGenerateNormalDouble(rng, B, length(B), mean, stddev))
#     return A
# end

# # log-normal
# const LognormalType = Union{Type{Float32},Type{Float64}}
# const LognormalArray = DenseCuArray{<:Union{Float32,Float64}}
# function rand_logn!(rng::RNG, A::DenseCuArray{Float32}; mean=0, stddev=1)
#     update_stream(rng)
#     inplace_pow2(A, B -> curandGenerateLogNormal(rng, B, length(B), mean, stddev))
#     return A
# end
# function rand_logn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
#     update_stream(rng)
#     inplace_pow2(A, B -> curandGenerateLogNormalDouble(rng, B, length(B), mean, stddev))
#     return A
# end

# # poisson
# const PoissonType = Union{Type{Cuint}}
# const PoissonArray = DenseCuArray{Cuint}
# function rand_poisson!(rng::RNG, A::DenseCuArray{Cuint}; lambda=1)
#     update_stream(rng)
#     curandGeneratePoisson(rng, A, length(A), lambda)
#     return A
# end

# # CPU arrays
# function Random.rand!(rng::RNG, A::AbstractArray{T}) where {T<:UniformType}
#     B = CuArray{T}(undef, size(A))
#     rand!(rng, B)
#     copyto!(A, B)
# end
# function Random.randn!(rng::RNG, A::AbstractArray{T}) where {T<:NormalType}
#     B = CuArray{T}(undef, size(A))
#     randn!(rng, B)
#     copyto!(A, B)
# end
# function rand_logn!(rng::RNG, A::AbstractArray{T}) where {T<:LognormalType}
#     B = CuArray{T}(undef, size(A))
#     rand_logn!(rng, B)
#     copyto!(A, B)
# end
# function rand_poisson!(rng::RNG, A::AbstractArray{T}) where {T<:PoissonType}
#     B = CuArray{T}(undef, size(A))
#     rand_poisson!(rng, B)
#     copyto!(A, B)
# end


# # GPU arrays
# Random.rand(rng::RNG, T::UniformType, dims::Dims) =
#     Random.rand!(rng, CuArray{T}(undef, dims))
# Random.randn(rng::RNG, T::NormalType, dims::Dims; kwargs...) =
#     outofplace_pow2(dims, shape -> CuArray{T}(undef, dims), A -> randn!(rng, A; kwargs...))
# rand_logn(rng::RNG, T::LognormalType, dims::Dims; kwargs...) =
#     outofplace_pow2(dims, shape -> CuArray{T}(undef, dims), A -> rand_logn!(rng, A; kwargs...))
# rand_poisson(rng::RNG, T::PoissonType, dims::Dims; kwargs...) =
#     rand_poisson!(rng, CuArray{T}(undef, dims); kwargs...)

# # specify default types
# Random.rand(rng::RNG, dims::Dims; kwargs...) = rand(rng, Float32, dims; kwargs...)
# Random.randn(rng::RNG, dims::Dims; kwargs...) = randn(rng, Float32, dims; kwargs...)
# rand_logn(rng::RNG, dims::Dims; kwargs...) = rand_logn(rng, Float32, dims; kwargs...)
# rand_poisson(rng::RNG, dims::Dims; kwargs...) = rand_poisson(rng, Cuint, dims; kwargs...)

# # support all dimension specifications
# Random.rand(rng::RNG, dim1::Integer, dims::Integer...) =
#     Random.rand(rng, Dims((dim1, dims...)))
# Random.randn(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
#     Random.randn(rng, Dims((dim1, dims...)); kwargs...)
# rand_logn(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
#     rand_logn(rng, Dims((dim1, dims...)); kwargs...)
# rand_poisson(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
#     rand_poisson(rng, Dims((dim1, dims...)); kwargs...)
# # ... and with a type
# Random.rand(rng::RNG, T::UniformType, dim1::Integer, dims::Integer...) =
#     Random.rand(rng, T, Dims((dim1, dims...)))
# Random.randn(rng::RNG, T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
#     Random.randn(rng, T, Dims((dim1, dims...)); kwargs...)
# rand_logn(rng::RNG, T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
#     rand_logn(rng, T, Dims((dim1, dims...)); kwargs...)
# rand_poisson(rng::RNG, T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
#     rand_poisson(rng, T, Dims((dim1, dims...)); kwargs...)

# # scalars
# Random.rand(rng::RNG, T::UniformType=Float32) = rand(rng, T, 1)[]
# Random.randn(rng::RNG, T::NormalType=Float32; kwargs...) = randn(rng, T, 1; kwargs...)[]
# rand_logn(rng::RNG, T::LognormalType=Float32; kwargs...) = rand_logn(rng, T, 1; kwargs...)[]
# rand_poisson(rng::RNG, T::PoissonType=Float32; kwargs...) = rand_poisson(rng, T, 1; kwargs...)[]
