export PowBlock, power

"""
    PowBlock{D,GT<:AbstractBlock} <: AbstractContainer{GT,D}

Repeat the same block `content` `pow` times
"""
struct PowBlock{D,BT<:AbstractBlock} <: AbstractContainer{BT,D}
    content::BT
    pow::Int
end

function PowBlock(content::BT, pow::Integer) where {D,BT<:AbstractBlock{D}}
    if pow < 0 && !isunitary(content)
        throw(ArgumentError("negative power requires a unitary block"))
    end
    PowBlock{D,BT}(content, Int(pow))
end

Yao.nqudits(pb::PowBlock) = nqudits(pb.content)
Yao.chsubblocks(pb::PowBlock, blk::AbstractBlock) = PowBlock(blk, pb.pow)
Yao.occupied_locs(pb::PowBlock) = occupied_locs(pb.content)

function Yao.mat(::Type{T}, pb::PowBlock{D}) where {T,D}
    pb.pow == 0 && return IMatrix{T}(D^nqudits(pb.content))
    pb.pow  > 0 && return mat(T, pb.content)^pb.pow
    return mat(T, adjoint(pb.content))^(-pb.pow)  # unitary: U^(-n) = (U†)^n
end

function Yao.unsafe_apply!(r::AbstractRegister, pb::PowBlock)
    blk = pb.pow >= 0 ? pb.content : adjoint(pb.content)
    for _ in 1:abs(pb.pow)
        Yao.unsafe_apply!(r, blk)
    end
    return r
end

Base.adjoint(pb::PowBlock) = PowBlock(adjoint(pb.content), pb.pow)
Base.:(==)(a::PowBlock, b::PowBlock) = a.pow == b.pow && a.content == b.content
Base.copy(pb::PowBlock) = PowBlock(pb.pow, pb.content)
Yao.cache_key(pb::PowBlock) = (pb.pow, cache_key(pb.content))
Yao.YaoBlocks.PropertyTrait(::PowBlock) = Yao.YaoBlocks.PreserveAll()
Yao.YaoBlocks.print_block(io::IO, pb::PowBlock) = print(io, "pow(", pb.pow, ")")

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
power(block::AbstractBlock, pow::Int) = PowBlock(block, pow)
power(block::AbstractBlock) = @λ(pow -> power(block, pow))

Base.:(^)(x::AbstractBlock, n::Int) = PowBlock(x, n)