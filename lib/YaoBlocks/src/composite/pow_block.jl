export PowBlock, power

using LinearAlgebra

"""
    PowBlock{D,GT<:AbstractBlock,PT<:Real} <: AbstractContainer{GT,D}
    PowBlock(block, pow) -> PowBlock

Repeat the same block `content` `pow` times
"""
struct PowBlock{D,BT<:AbstractBlock,PT<:Real} <: AbstractContainer{BT,D}
    content::BT
    pow::PT
end

function PowBlock(content::BT, pow::PT) where {D,BT<:AbstractBlock{D},PT<:Real}
    if pow isa Int && pow < 0 && !isunitary(content)
        throw(ArgumentError("negative power requires a unitary block"))
    end
    PowBlock{D,BT,PT}(content, pow)
end

nqudits(pb::PowBlock) = nqudits(pb.content)
chsubblocks(pb::PowBlock, blk::AbstractBlock) = PowBlock(blk, pb.pow)
occupied_locs(pb::PowBlock) = occupied_locs(pb.content)

function mat(::Type{T}, pb::PowBlock{D}) where {T,D}
    pb.pow == 0 && return IMatrix{T}(D^nqudits(pb.content))
    pb.pow  > 0 && return mat(T, pb.content)^pb.pow
    return mat(T, adjoint(pb.content))^(-pb.pow)  # unitary: U^(-n) = (U†)^n
end

function YaoAPI.unsafe_apply!(r::AbstractRegister, pb::PowBlock{D, BT, PT}) where {D, BT<:AbstractBlock,PT<:Integer}
    blk = pb.pow >= 0 ? pb.content : adjoint(pb.content)
    for _ in 1:abs(pb.pow)
        YaoAPI.unsafe_apply!(r, blk)
    end
    return r
end

# Fall back for non integer powers
function YaoAPI.unsafe_apply!(r::AbstractRegister, pb::PowBlock)
    YaoAPI.unsafe_apply!(r, matblock(mat(pb)))
end

function nparameters(pb::PowBlock)
    return iszero(pb.pow) ? 0 : nparameters(pb.content)
end
Base.adjoint(pb::PowBlock) = PowBlock(adjoint(pb.content), pb.pow)
Base.:(==)(a::PowBlock, b::PowBlock) = a.pow == b.pow && a.content == b.content
Base.copy(pb::PowBlock) = PowBlock(pb.pow, pb.content)
cache_key(pb::PowBlock) = (pb.pow, cache_key(pb.content))
PropertyTrait(::PowBlock) = PreserveAll()

"""
    power(pow::Int, block::AbstractBlock) -> PowBlock
    power(block::AbstractBlock) -> pow -> PowBlock

Create a [`PowBlock`](@ref) that applies `block` exactly `n` times.
The lazy form `power(n)` returns a function that constructs the block when called.

### Examples

```julia
power(3, X)           # applies X three times
power(Rx(0.5))(3)     # lazy form
```
"""
power(block::AbstractBlock, pow::Real) = PowBlock(block, pow)
