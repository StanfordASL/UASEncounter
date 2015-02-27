
module SVDSHack

import Base.size
import Base.*
import Base.issym

export svds

## svds

type SVDOperator{T,S} <: AbstractArray{T, 2}
    X::S
    m::Int
    n::Int
    SVDOperator(X::S) = new(X, size(X,1), size(X,2))
end

## v = [ left_singular_vector; right_singular_vector ]
*{T,S}(s::SVDOperator{T,S}, v::Vector{T}) = [s.X * v[s.m+1:end]; s.X' * v[1:s.m]]
size(s::SVDOperator)  = s.m + s.n, s.m + s.n
issym(s::SVDOperator) = true

function svds{S}(X::S; nsv::Int = 6, ritzvec::Bool = true, tol::Float64 = 0.0, maxiter::Int = 1000)
    if nsv < 1
        throw(ArgumentError("number of singular values (nsv) must be ≥ 1, got $nsv"))
    end
    if nsv > minimum(size(X))
        throw(ArgumentError("number of singular values (nsv) must be ≤ $(minimum(size(X))), got $nsv"))
    end
    otype = eltype(X)
    # run(`free -m`)
    # println("\n\n\n=========\n\n\n")
    ex    = eigs(SVDOperator{otype,S}(X), I; ritzvec = ritzvec, nev = 2*nsv, tol = tol, maxiter = maxiter)
    # ex    = eigs(SVDOperator{otype,S}(X), I; ritzvec = ritzvec, nev = 2*nsv, tol = tol, maxiter = maxiter, which=:SM)
    # run(`free -m`)
    ind   = [1:2:nsv*2]
    sval  = abs(ex[1][ind])

    ritzvec || return (sval, ex[2], ex[3], ex[4], ex[5])

    # calculating singular vectors
    left_sv  = sqrt(2) * ex[2][ 1:size(X,1),     ind ] .* sign(ex[1][ind]')
    right_sv = sqrt(2) * ex[2][ size(X,1)+1:end, ind ]
    return (left_sv, sval, right_sv, ex[3], ex[4], ex[5], ex[6])
end

end
