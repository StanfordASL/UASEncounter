
module EncounterValueIteration

using EncounterModel: IntruderParams, OwnshipParams, SimParams, EncounterState, EncounterAction, ownship_dynamics, encounter_dynamics, next_state_from_pd, post_decision_state, reward, SIM
using EncounterFeatures: AssembledFeatureBlock
using EncounterSimulation: LinearQValuePolicy
import SVDSHack

export run_sims, iterate, extract_policy

function gen_state(rng::AbstractRNG)
    ix = 1000.0*rand(rng)
    iy = 2000.0*(rand(rng)-0.5)
    ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))
    # ix = -1000
    # iy = 1000
    # ihead = pi/2.0

    ox = 1000.0*rand(rng)
    oy = 1000.0*(rand(rng)-0.5)
    ohead = 2*pi*rand(rng) 

    return EncounterState([ox,oy,ohead],[ix, iy, ihead],false)
end

function run_sims(new_phi::AssembledFeatureBlock, phi::AssembledFeatureBlock, lambda::AbstractVector{Float64}, actions, N::Int, NEV::Int, rng_seed::Int)
    has_data = Set{Int64}()
    phis = Any[]
    v = Array(Float64, N)
    rng = MersenneTwister(rng_seed)

    # XXX
    # gc()
    # gc_disable()

    for n in 1:N
        sn = gen_state(rng)

        # v_sums = Array(Float64, length(ACTIONS))
        v_sums = zeros(length(actions))

        for m in 1:length(actions)
            a = actions[m]
            pd = post_decision_state(sn, a)

            for l in 1:NEV
                sp = next_state_from_pd(pd, rng)
                if !sp.end_state
                    v_sums[m] += sum(phi.features(sp)'*lambda) # sum is just because that's the easiest way to convert to a float
                end
            end
        end

        # # XXX
        # if mod(n, 50) == 0
        #     gc_enable()
        #     gc()
        #     gc_disable()
        # end

        v[n] = reward(sn) + maximum(v_sums)/NEV
        # v[n] = maximum(v_sums)/NEV

        feat = new_phi.features(sn)
        push!(phis, feat)

        if new_phi.dense
            J = find(feat)        
        else
            (J, dummy, dummy2) = findnz(feat)
        end
        union!(has_data, J)
    end

    return (phis, has_data, v)
end

function iterate{A<:EncounterAction}(phi::AssembledFeatureBlock, lambda::AbstractVector{Float64}, actions::Vector{A}, num_sims::Int; new_phi=nothing, num_EV::Int=20, rng_seed_offset::Int=0, sims_per_spawn::Int=1000, convert_to_sparse=false, parallel=true)
    if new_phi==nothing
        new_phi = phi
    end
    new_lambda = Array(Float64, new_phi.length)

    has_data = Set{Int64}()
    phirows = Any[]
    v = Float64[]

    println("spawning simulations...")
    refs = Any[]
    for i in 1:int(num_sims/sims_per_spawn)
        if parallel
            push!(refs, @spawn run_sims(new_phi, phi, lambda, actions, sims_per_spawn, num_EV, rng_seed_offset+i))
        else
            push!(refs, run_sims(new_phi, phi, lambda, actions, sims_per_spawn, num_EV, rng_seed_offset+i))
            println("Finished batch $i")
        end
    end

    println("waiting for sims to finish and aggregating simulation data...")

    tic()
    num_fetched = 0
    for ref in refs
        (local_phis, local_has_data, local_v) = fetch(ref)
        append!(phirows, local_phis)
        append!(v, local_v)
        union!(has_data, local_has_data)
        num_fetched += sims_per_spawn
        print("\rfetched $num_fetched simulation results")
    end
    print("\n")
    toc()

    println("done with simulations.")

    @everywhere gc()

    println("building matrix...")
    tic()

    if new_phi.dense && !convert_to_sparse
        println("Phi is dense ($num_sims x $(length(new_lambda)))")
        Phi = Array(Float64, num_sims, length(new_lambda))
        for n in 1:num_sims
            Phi[n,:] = phirows[n]
        end
    else
        println("Phi is sparse ($num_sims x $(length(has_data)))")

        full_to_small = Array(Int64, length(new_lambda))
        small_to_full = Array(Int64, length(has_data))
        j = 1
        for i in 1:length(new_lambda)
            if i in has_data
                full_to_small[i] = j
                small_to_full[j] = i
                j+=1
            else
                full_to_small[i] = -1
            end
        end

        Phi = spzeros(num_sims, length(has_data))

        for i in 1:length(phirows)
            J = Int[]
            V = Float64[]
            if new_phi.dense
                J = find(phirows[i])
                V = phirows[i][J]
            else
                (J, dummy, V) = findnz(phirows[i])
            end
            for k in 1:length(J)
                @assert(full_to_small[J[k]] > 0)
                Phi[i,full_to_small[J[k]]] = V[k]
            end
        end
    end

    refs=nothing
    phirows=nothing
    @everywhere gc()
    toc()

    println("inverting...")
    tic()

    if new_phi.dense && !convert_to_sparse
        new_lambda = pinv(Phi)*v
        if length(new_lambda) <= 100
            @show new_lambda
        end
    else
        svd = SVDSHack.svds(Phi, nsv = min(length(has_data),200), tol = 0.1)
        left_sv = svd[1]
        sval = svd[2]
        println("singular values: $sval")
        right_sv = svd[3]
        sinv = 1 ./ sval
        tmp = scale(sinv, left_sv')*v
        lambda_est = right_sv*tmp
    end
    toc()

    if !new_phi.dense
        println("filling...")
        new_lambda[small_to_full] = lambda_est
        need_data = setdiff!(Set{Int64}(1:length(new_lambda)), has_data)
        for i in need_data
            new_lambda[i] = lambda[i]
        end
    end
    # fill_r!(new_r, need_data)

#     println("filling...")
# 
#     new_r[small_to_full] = r_est
#     need_data = setdiff!(Set{Int64}(1:length(new_r)), has_data)
#     fill_r!(new_r, need_data)

    return new_lambda
end

function extract_policy(phi::AssembledFeatureBlock, lambda::AbstractVector{Float64}, actions::Vector{EncounterAction}, num_sims::Int; new_phi=nothing, num_EV::Int=20)
    if new_phi == nothing
        new_phi = phi
    end

    p = LinearQValuePolicy(new_phi, actions, Array(Vector{Float64}, length(actions)))

    for i in 1:length(p.actions)
        p.lambdas[i] = iterate(phi, lambda, [p.actions[i]], num_sims; new_phi=new_phi, num_EV=num_EV)
    end

    return p
end

end
