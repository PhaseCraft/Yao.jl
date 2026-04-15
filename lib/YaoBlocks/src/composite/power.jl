export Power, power

using LinearAlgebra

"""
    Power{D,GT<:AbstractBlock,PT<:Real} <: AbstractContainer{GT,D}
    Power(block, pow) -> Power

Repeat the same block `content` `pow` times
"""
struct Power{D,BT<:AbstractBlock,PT<:Real} <: AbstractContainer{BT,D}
    content::BT
    pow::PT
end

function Power(content::BT, pow::PT) where {D,BT<:AbstractBlock{D},PT<:Real}
    if pow isa Int && pow < 0 && !isunitary(content)
        throw(ArgumentError("negative power requires a unitary block"))
    end
    Power{D,BT,PT}(content, pow)
end

nqudits(pb::Power) = nqudits(pb.content)
chsubblocks(pb::Power, blk::AbstractBlock) = Power(blk, pb.pow)
occupied_locs(pb::Power) = occupied_locs(pb.content)

function mat(::Type{T}, pb::Power{D}) where {T,D}
    pb.pow == 0 && return IMatrix{T}(D^nqudits(pb.content))
    pb.pow  > 0 && return mat(T, pb.content)^pb.pow
    return mat(T, adjoint(pb.content))^(-pb.pow)  # unitary: U^(-n) = (U†)^n
end

function YaoAPI.unsafe_apply!(r::AbstractRegister, pb::Power{D, BT, PT}) where {D, BT<:AbstractBlock,PT<:Integer}
    blk = pb.pow >= 0 ? pb.content : adjoint(pb.content)
    for _ in 1:abs(pb.pow)
        YaoAPI.unsafe_apply!(r, blk)
    end
    return r
end

# Fall back for non integer powers
function YaoAPI.unsafe_apply!(r::AbstractRegister, pb::Power)
    YaoAPI.unsafe_apply!(r, matblock(mat(pb)))
end

function nparameters(pb::Power)
    return iszero(pb.pow) ? 0 : nparameters(pb.content)
end
Base.adjoint(pb::Power) = Power(adjoint(pb.content), pb.pow)
Base.:(==)(a::Power, b::Power) = a.pow == b.pow && a.content == b.content
Base.copy(pb::Power) = Power(pb.pow, pb.content)
cache_key(pb::Power) = (pb.pow, cache_key(pb.content))
PropertyTrait(::Power) = PreserveAll()
