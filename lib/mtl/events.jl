export MtlEvent, MtlSharedEvent, MtlSharedEventHandle

abstract type MtlAbstractEvent end

const MTLEvent = Ptr{MtEvent}
const MTLSharedEvent = Ptr{MtSharedEvent}
const MTLSharedEventHandle = Ptr{MtSharedEventHandle}

mutable struct MtlEvent <: MtlAbstractEvent
	handle::MTLEvent
	device::MtlDevice
end

mutable struct MtlSharedEvent <: MtlAbstractEvent
	handle::MTLSharedEvent
	device::MtlDevice
end

Base.unsafe_convert(::Type{MTLEvent}, ev::MtlAbstractEvent) = convert(MTLEvent, ev.handle)
Base.unsafe_convert(::Type{MTLSharedEvent}, ev::MtlSharedEvent) = ev.handle

Base.:(==)(a::MtlAbstractEvent, b::MtlAbstractEvent) = a.handle == b.handle
Base.hash(ev::MtlAbstractEvent, h::UInt) = hash(ev.handle, h)

function unsafe_destroy!(fun::MtlAbstractEvent)
	mtRelease(fun.handle)
end

function MtlEvent(dev::MtlDevice)
	handle = mtDeviceNewEvent(dev)
	obj = MtlEvent(handle, dev)
	finalizer(unsafe_destroy!, obj)
	return obj
end

function MtlSharedEvent(dev::MtlDevice)
	handle = mtDeviceNewSharedEvent(dev)
	obj = MtlSharedEvent(handle, dev)
	finalizer(unsafe_destroy!, obj)
	return obj
end


## properties

Base.propertynames(::MtlAbstractEvent) = (:device, :label, :signaledValue)

function Base.getproperty(ev::MtlAbstractEvent, f::Symbol)
    if f === :label
        ptr = mtEventLabel(ev)
        ptr == C_NULL ? nothing : unsafe_string(ptr)
    elseif ev isa MtlSharedEvent && f === :signaledValue
        mtSharedEventSignaledValue(ev)
    else
        getfield(ev, f)
    end
end

function Base.setproperty!(ev::MtlAbstractEvent, f::Symbol, val)
    if f === :label
		mtEventLabelSet(ev, val)
    else
        setfield!(ev, f, val)
    end
end


## shared event handle

mutable struct MtlSharedEventHandle
	handle::MTLSharedEventHandle
	event::MtlSharedEvent
end

function MtlSharedEventHandle(ev::MtlSharedEvent)
	handle = mtSharedEventNewHandle(ev)
	obj = MtlSharedEventHandle(handle, ev)
	finalizer(unsafe_destroy!, obj)
	return obj
end

function unsafe_destroy!(evh::MtlSharedEventHandle)
	mtRelease(evh.handle)
end

Base.unsafe_convert(::Type{MTLSharedEventHandle}, evh::MtlSharedEventHandle) = evh.handle

Base.:(==)(a::MtlSharedEventHandle, b::MtlSharedEventHandle) = a.handle == b.handle
Base.hash(evh::MtlSharedEventHandle, h::UInt) = hash(evh.handle, h)

function wait(ev::MtlSharedEvent, val)
	mtSharedEventWait(ev, val)
end

function signal(ev::MtlSharedEvent, val)
	mtSharedEventSignal(ev, val)
end

function isdone(ev::MtlAbstractEvent, val)
    if ev.signaledValue >= val
        return true
    else
        return false
    end
end

function wait(ev::MtlSharedEvent, val)
    # perform as much of the sync as possible without blocking in Metal.
    # XXX: remove this using a yield callback, or by synchronizing on a dedicated thread?
    nonblocking_synchronize(ev, val)

	mtSharedEventWait(ev, val)
end

@inline function cooperative_wait(ev::MtlSharedEvent, val)
    # fast path
    isdone(ev, val) && return

    # spin (initially without yielding to minimize latency)
    spins = 0
    while spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # Temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        isdone(ev, val) && return
        spins += 1
    end

    return
end