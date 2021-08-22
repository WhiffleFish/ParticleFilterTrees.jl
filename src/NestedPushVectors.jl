#=
Base julia sizehint! + push! garbage compared to
resize! + setindex!

Remove when sizehinting in base julia isn't garbage
=#

mutable struct NestedPushVector{T, K<:PushVector{T}, V<:AbstractVector{K}} <: AbstractVector{K}
    "Vector used for storage."
    parent::V
    "Number of elements held by `parent`."
    len::Int
    "Inner sizehint"
    inner_sh::Int
    "Outer sizehint"
    outer_sh::Int
end

# Simplifications
const PV{T} = PushVector{T, Vector{T}} where {T}
const NPV{T} = NestedPushVector{T, PV{T}, Vector{PV{T}}} where {T}

function NestedPushVector{T}(inner_sh::Integer = 4, outer_sh::Integer = 4) where {T}
    inner_sh ≥ 0 && outer_sh ≥ 0 || throw(DomainError((inner_sh, outer_sh), "Invalid initial size."))
    NestedPushVector(
        [PushVector{T}(inner_sh) for _ in 1:outer_sh],
        0,
        inner_sh,
        outer_sh
    )
end

@inline Base.length(v::NestedPushVector) = v.len
@inline Base.size(v::NestedPushVector) = (v.len, )

function extend!(v::NestedPushVector{T}, n::Int) where {T}
    ish = v.inner_sh
    prev_l = length(v.parent)
    resize!(v.parent, prev_l + n)
    for i in 1:n
        v.parent[prev_l + i] = PushVector{T}(ish)
    end
    nothing
end

@inline function Base.getindex(v::NestedPushVector, i)
    @boundscheck checkbounds(v, i)
    @inbounds v.parent[i]
end

@inline function Base.setindex!(v::NestedPushVector, x::PushVector, i)
    @boundscheck checkbounds(v, i)
    @inbounds v.parent[i] = x
end

function freenext!(v::NestedPushVector)
    v.len += 1
    if v.len > length(v.parent)
        extend!(v, v.len)
    end
    nothing
end

function Base.push!(v::NestedPushVector, x::PushVector)
    v.len += 1
    if v.len > length(v.parent)
        extend!(v, v.len)
    end
    v.parent[v.len] = x
    v
end

function Base.empty!(v::NestedPushVector)
    resize!(v.parent, v.outer_sh)
    v.len = 0
    for pv in v.parent
        resize!(pv.parent, v.inner_sh)
        pv.len = 0
    end
    nothing
end
