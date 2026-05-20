export map_address, unsafe_map_address, AddressInfo

struct AddressInfo
    nbits::Int
    addresses::Vector{Int}
end
AddressInfo(nbits::Int, ::AllLocs) = AddressInfo(nbits, collect(1:nbits))
AddressInfo(nbits::Int, iter) = AddressInfo(nbits, collect(iter))
Base.copy(info::AddressInfo) = AddressInfo(copy(info.addresses))
Base.:/(locs, info::AddressInfo) = map(loc -> info.addresses[loc], locs)
Base.:/(locs::AllLocs, info::AddressInfo) = info.addresses

"""
    map_address(block::AbstractBlock, info::AddressInfo) -> AbstractBlock

map the locations in `block` to target locations.

# Example

`map_address` can be used to embed a sub-circuit to a larger one.

```jldoctest; setup=:(using YaoBlocks)
julia> c = chain(5, repeat(H, 1:5), put(2=>X), kron(1=>X, 3=>Y))
nqubits: 5
chain
├─ repeat on (1, 2, 3, 4, 5)
│  └─ H
├─ put on (2)
│  └─ X
└─ kron
   ├─ 1=>X
   └─ 3=>Y


julia> map_address(c, AddressInfo(10, [6,7,8,9,10]))
nqubits: 10
chain
├─ repeat on (6, 7, 8, 9, 10)
│  └─ H
├─ put on (7)
│  └─ X
└─ kron
   ├─ 6=>X
   └─ 8=>Y
```
"""
function map_address end

function map_address(block::AbstractBlock, info::AddressInfo)
    throw(NotImplementedError(:map_address, typeof(block)))
end

function map_address(blk::Measure{D}, info::AddressInfo) where D
    m = Measure{D}(info.nbits,
        blk.rng,
        blk.operator,
        (blk.locations / info...,),
        blk.postprocess,
        blk.error_prob,
    )
    if isdefined(blk, :results)
        m.results = blk.results
    end
    return m
end

function map_address(blk::PrimitiveBlock, info::AddressInfo)
    @assert length(info.addresses) == nqudits(blk) "expected $(nqudits(blk)) addresses, got $(length(info.addresses))"
    if length(info.addresses) == info.nbits
        return blk
    else
        # raise the number of qubits
        return put(info.nbits, info.addresses => blk)
    end
end
map_address(blk::PutBlock, info::AddressInfo) =
    put(info.nbits, blk.locs / info => content(blk))

function map_address(blk::ControlBlock, info::AddressInfo)
    ControlBlock(info.nbits,
        blk.ctrl_locs / info,
        blk.ctrl_config,
        content(blk),
        blk.locs / info,
    )
end

function map_address(blk::KronBlock, info::AddressInfo)
    kron(info.nbits, [l => G for (l, G) in zip(blk.locs / info, blk.blocks)]...)
end

function map_address(blk::RepeatedBlock, info::AddressInfo)
    repeat(info.nbits, content(blk), blk.locs / info)
end

function map_address(blk::Subroutine, info::AddressInfo)
    subroutine(info.nbits, content(blk), blk.locs / info)
end

function map_address(blk::ChainBlock, info::AddressInfo)
    chain(info.nbits, map(b -> map_address(b, info), subblocks(blk)))
end

function map_address(blk::Daggered, info::AddressInfo)
    Daggered(map_address(content(blk), info))
end

function map_address(blk::CachedBlock, info::AddressInfo)
    CachedBlock(blk.server, map_address(content(blk), info), blk.level)
end

function map_address(blk::Scale, info::AddressInfo)
    Scale(blk.alpha, map_address(content(blk), info))
end

function map_address(blk::AbstractAdd, info::AddressInfo)
    chsubblocks(blk, map(b -> map_address(b, info), subblocks(blk)))
end

function map_address(blk::Power, info::AddressInfo)
    Power(map_address(content(blk), info), blk.pow)
end

"""
    unsafe_map_address(block::AbstractBlock, info::AddressInfo) -> AbstractBlock

Like [`map_address`](@ref) but uses unsafe constructors that skip validity checks.
"""
function unsafe_map_address end

function unsafe_map_address(block::AbstractBlock, info::AddressInfo)
    throw(NotImplementedError(:unsafe_map_address, typeof(block)))
end

function unsafe_map_address(blk::Measure{D}, info::AddressInfo) where D
    m = Measure{D}(info.nbits,
        blk.rng,
        blk.operator,
        (blk.locations / info...,),
        blk.postprocess,
        blk.error_prob,
    )
    if isdefined(blk, :results)
        m.results = blk.results
    end
    return m
end

function unsafe_map_address(blk::PrimitiveBlock, info::AddressInfo)
    if length(info.addresses) == info.nbits
        return blk
    else
        return put(info.nbits, info.addresses => blk)
    end
end

unsafe_map_address(blk::PutBlock, info::AddressInfo) =
    put(info.nbits, blk.locs / info => content(blk))

function unsafe_map_address(blk::ControlBlock, info::AddressInfo)
    new_ctrl_locs = map((l, c) -> c == 1 ? l : -l, blk.ctrl_locs / info, blk.ctrl_config)
    unsafe_control(info.nbits, new_ctrl_locs, blk.locs / info => content(blk))
end

function unsafe_map_address(blk::KronBlock, info::AddressInfo)
    mapped = blk.locs / info
    unsafe_kron(info.nbits, (first(l):last(l) => b for (l, b) in zip(mapped, blk.blocks))...)
end

function unsafe_map_address(blk::RepeatedBlock, info::AddressInfo)
    repeat(info.nbits, content(blk), blk.locs / info)
end

function unsafe_map_address(blk::Subroutine, info::AddressInfo)
    unsafe_subroutine(info.nbits, content(blk), blk.locs / info)
end

function unsafe_map_address(blk::ChainBlock{D}, info::AddressInfo) where D
    new_blocks = AbstractBlock{D}[unsafe_map_address(b, info) for b in subblocks(blk)]
    unsafe_chain(info.nbits, new_blocks)
end

function unsafe_map_address(blk::Daggered, info::AddressInfo)
    Daggered(unsafe_map_address(content(blk), info))
end

function unsafe_map_address(blk::CachedBlock, info::AddressInfo)
    CachedBlock(blk.server, unsafe_map_address(content(blk), info), blk.level)
end

function unsafe_map_address(blk::Scale, info::AddressInfo)
    Scale(blk.alpha, unsafe_map_address(content(blk), info))
end

function unsafe_map_address(blk::AbstractAdd, info::AddressInfo)
    chsubblocks(blk, map(b -> unsafe_map_address(b, info), subblocks(blk)))
end

function unsafe_map_address(blk::Power, info::AddressInfo)
    Power(unsafe_map_address(content(blk), info), blk.pow)
end
