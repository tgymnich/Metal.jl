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


function MPSMatrixRandomDistributionDescriptor()
    desc = @objc [MPSMatrixRandomDistributionDescriptor defaultDistributionDescriptor]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixRandomDistributionDescriptor(minimum, maximum)
    desc = @objc [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:minimum::Cfloat
                                      maximum:maximum::Cfloat]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixRandomDistributionDescriptor(mean, standardDeviation)
    desc = @objc [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:mean::Cfloat
                                      standardDeviation:standardDeviation::Cfloat]::id{MPSMatrixRandomDistributionDescriptor}
    obj = MPSMatrixRandomDistributionDescriptor(desc)
    # XXX: who releases this object?
    return obj
end

function MPSMatrixRandomDistributionDescriptor(mean, standardDeviation, minimum, maximum)
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

"""
Generates random numbers using a Mersenne Twister algorithm suitable for GPU execution.
It uses a period of 2**11214. 
For further details see: Mutsuo Saito. A Variant of Mersenne Twister Suitable for Graphic Processors. arXiv:1005.4973
"""
@objcwrapper immutable=false MPSMatrixRandomMTGP32 <: MPSMatrixRandom


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

"""
Generates random numbers using a counter based algorithm. 
For further details see:
John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel Random Numbers: As Easy as 1, 2, 3.
"""
@objcwrapper immutable=false MPSMatrixRandomPhilox <: MPSMatrixRandom


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
    obj = MPSMatrixRandomMTGP32(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixRandomPhilox} initWithDevice:device::id{MTLDevice}
                                          destinationDataType:destinationDataType::MPSDataType
                                          seed:seed::NSUInteger
                                          distributionDescriptor:distributionDescriptor::id{MPSMatrixRandomDistributionDescriptor}]::id{MPSMatrixRandomPhilox}
    return obj
end