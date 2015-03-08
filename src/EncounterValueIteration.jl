
module EncounterValueIteration

using EncounterModel: IntruderParams, OwnshipParams, SimParams, EncounterState, EncounterAction, RewardModel, ownship_dynamics, encounter_dynamics, next_state_from_pd, post_decision_state, reward, SIM
using EncounterFeatures: FeatureBlock, f_focused_intruder_grid, evaluate
using EncounterSimulation: LinearQValuePolicy
using GridInterpolations
import SVDSHack

export run_sims, iterate, extract_policy, gen_state_snap_to_grid, gen_ic_batch_for_grid, find_policy

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

    has_deviated = rand(rng) >= 0.5

    return EncounterState([ox,oy,ohead],[ix, iy, ihead],false,has_deviated)
end

# HACK
function gen_state_snap_to_grid(rng::AbstractRNG, grid)
    state=gen_state(rng)
    
    is = state.is
    os = state.os

    d = norm(os[1:2]-is[1:2])-SIM.legal_D

    # bearing to ownship from intruder's perspective
    bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    while bearing >= pi bearing -= 2*pi end
    while bearing < -pi bearing += 2*pi end
    # relative heading of ownship compared to intruder
    heading = os[3] - is[3]
    while heading >= 2*pi heading -= 2*pi end
    while heading < 0.0 heading += 2*pi end

    if bearing > pi/2 || bearing < -pi/2 || d <= 0.0 || d > maximum(grid.cutPoints[1])
        return state
    end

    inds, weights = interpolants(grid, [d, bearing, heading])

    # @show ind2x(grid, inds[indmax(weights)])
    (dnew,bnew,hnew) = ind2x(grid, inds[indmax(weights)])
    if dnew == 0.0
        dnew += 1e-5
    end

    state.os[1:2] = is[1:2] + [(dnew+SIM.legal_D)*cos(is[3]+bnew), (dnew+SIM.legal_D)*sin(is[3]+bnew)]
    state.os[3] = hnew+is[3]

    # DEBUG
    # is = state.is
    # os = state.os

    # d = norm(os[1:2]-is[1:2])-SIM.legal_D

    # # bearing to ownship from intruder's perspective
    # bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    # while bearing >= pi bearing -= 2*pi end
    # while bearing < -pi bearing += 2*pi end
    # # relative heading of ownship compared to intruder
    # heading = os[3] - is[3]
    # while heading >= 2*pi heading -= 2*pi end
    # while heading < 0.0 heading += 2*pi end

    # @show (dnew,d)
    # @show (bnew,bearing)
    # @show (hnew,heading)

    return state
end

# HACK
function gen_ic_batch_for_grid(rng, grid)
    ics = Array(EncounterState, length(grid))
    for i in 1:length(grid)
        ix = 1000.0*rand(rng)
        iy = 2000.0*(rand(rng)-0.5)
        ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))
        is = [ix, iy, ihead]
        (dnew,bnew,hnew) = ind2x(grid,i)
        if dnew <= 0.0
            dnew+=1e-5
        end
        if bnew > pi/2 - 1e-5
            bnew-=1e-5
        end
        if bnew < -pi/2 + 1e-5
            bnew+=1e-5
        end
        os = [is[1]+(dnew+SIM.legal_D)*cos(is[3]+bnew),
              is[2]+(dnew+SIM.legal_D)*sin(is[3]+bnew),
              is[3]+hnew]
        ics[i] = EncounterState(os, is, false)
        feat = f_focused_intruder_grid(ics[i],grid)
        if length(find(feat))==0
            @show (dnew,bnew,hnew)
        end
    end
    return ics
end

function run_sims(new_phi::FeatureBlock, phi::FeatureBlock, theta::AbstractVector{Float64}, rm::RewardModel, actions, N::Int, NEV::Int, rng_seed::Int; state_gen::Function=gen_state, ic_batch::Vector{EncounterState}=EncounterState[])
    has_data = Set{Int64}()
    phis = Any[]
    v = Array(Float64, N)
    rng = MersenneTwister(rng_seed)

    # XXX
    # gc()
    # gc_disable()

    for n in 1:N
        if n <= length(ic_batch)
            sn = ic_batch[n]
        else
            sn = state_gen(rng)
        end

        v_sums = zeros(length(actions))
        rewards = zeros(length(actions))

        for m in 1:length(actions)
            a = actions[m]
            pd = post_decision_state(sn, a)
            rewards[m] = reward(rm, sn, a)

            for l in 1:NEV
                sp = next_state_from_pd(pd, rng)
                if !sp.end_state
                    v_sums[m] += sum(evaluate(phi,sp)'*theta) # sum is just because that's the easiest way to convert to a float
                end
            end
        end

        # # XXX
        # if mod(n, 50) == 0
        #     gc_enable()
        #     gc()
        #     gc_disable()
        # end

        v[n] = maximum(rewards + v_sums/NEV)

        feat = evaluate(new_phi,sn)
        push!(phis, feat)

        # if new_phi.dense
        #     J = find(feat)        
        # else
        #     (J, dummy, dummy2) = findnz(feat)
        # end
        J = find(feat)        
        union!(has_data, J)
    end

    return (phis, has_data, v)
end

function find_policy{A<:EncounterAction}(phi::FeatureBlock,
                     rm::RewardModel,
                     actions::Vector{A},
                     intruder_grid::AbstractGrid)

    rng0 = MersenneTwister(0)

    theta = zeros(length(phi))
    snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid)

    for i in 1:30
        tic()
        sims_per_policy = 10000
        # println("starting value iteration $i ($sims_per_policy simulations)")
        ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
        theta_new = iterate(phi, theta, rm, actions, sims_per_policy,
                            rng_seed_offset=i,
                            state_gen=snap_generator,
                            parallel=true,
                            ic_batch=ic_batch,
                            output_prefix="\r[$i ($sims_per_policy)]",
                            output_suffix="")
        theta = theta_new
        toc()
    end

    tic()
    sims_per_policy = 50000
    # println("starting final value iteration ($sims_per_policy simulations)")
    ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
    theta_new = iterate(phi, theta, rm, actions, sims_per_policy,
                        rng_seed_offset=2011,
                        state_gen=snap_generator,
                        parallel=true,
                        ic_batch=ic_batch,
                        output_prefix="\r[final ($sims_per_policy)]",
                        output_suffix="")
    theta = theta_new
    toc()

    ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
    return extract_policy(phi, theta, rm, actions, 50000,
                            ic_batch=ic_batch,
                            state_gen=snap_generator)
end

function iterate{A<:EncounterAction}(phi::FeatureBlock,
                                     theta::AbstractVector{Float64},
                                     rm::RewardModel,
                                     actions::Vector{A},
                                     num_sims::Int;
                                     new_phi=nothing,
                                     num_EV::Int=20,
                                     rng_seed_offset::Int=0,
                                     sims_per_spawn::Int=1000,
                                     convert_to_sparse=false,
                                     parallel=true,
                                     state_gen::Function=gen_state,
                                     ic_batch::Vector{EncounterState}=EncounterState[],
                                     output_prefix="",
                                     output_suffix="\n")

    if new_phi==nothing
        new_phi = phi
    end
    new_theta = Array(Float64, length(new_phi))

    has_data = Set{Int64}()
    phirows = Any[]
    v = Float64[]

    print("$output_prefix spawning simulations... $output_suffix")
    refs = Any[]
    for i in 1:int(num_sims/sims_per_spawn)
        ic_batch_part=ic_batch[(i-1)*sims_per_spawn+1:min(i*sims_per_spawn,length(ic_batch))]
        if parallel
            push!(refs, @spawn run_sims(new_phi, phi, theta, rm, actions, sims_per_spawn, num_EV, rng_seed_offset+i, state_gen=state_gen, ic_batch=ic_batch_part))
        else
            push!(refs, run_sims(new_phi, phi, theta, rm, actions, sims_per_spawn, num_EV, rng_seed_offset+i, state_gen=state_gen, ic_batch=ic_batch_part))
            # println("Finished batch $i")
        end
    end

    print("$output_prefix waiting for sims to finish and aggregating simulation data... $output_suffix")

    # tic()
    num_fetched = 0
    for ref in refs
        (local_phis, local_has_data, local_v) = fetch(ref)
        append!(phirows, local_phis)
        append!(v, local_v)
        union!(has_data, local_has_data)
        num_fetched += sims_per_spawn
        print("\r$output_prefix fetched $num_fetched sims.")
    end
    # print("\n")
    # toc()

    print("$output_prefix done with sims. $output_suffix")

    @everywhere gc()

    print("$output_prefix building matrix... $output_suffix")
    # tic()

    print("$output_prefix Phi is dense ($num_sims x $(length(new_theta))) $output_suffix")
    Phi = Array(Float64, num_sims, length(new_theta))
    for n in 1:num_sims
        Phi[n,:] = phirows[n]
    end
    # @show rank(Phi)

    refs=nothing
    phirows=nothing
    @everywhere gc()
    # toc()

    print("$output_prefix inverting... $output_suffix")
    # tic()

    new_theta = pinv(Phi)*v
    if length(new_theta) <= 100
        @show new_theta
    end
    # toc()

    return new_theta
end

function extract_policy(phi::FeatureBlock,
                        theta::AbstractVector{Float64},
                        rm::RewardModel,
                        actions::Vector{EncounterAction},
                        num_sims::Int;
                        new_phi=nothing,
                        num_EV::Int=20,
                        state_gen::Function=gen_state, 
                        ic_batch::Vector{EncounterState}=EncounterState[])

    if new_phi == nothing
        new_phi = phi
    end

    p = LinearQValuePolicy(new_phi, actions, Array(Vector{Float64}, length(actions)))

    for i in 1:length(p.actions)
        p.thetas[i] = iterate(phi, theta, rm, [p.actions[i]], num_sims,
                              new_phi=new_phi,
                              num_EV=num_EV,
                              ic_batch=ic_batch,
                              state_gen=state_gen,
                              output_prefix="\r[$(p.actions[i]) ($num_sims)]",
                              output_suffix="")
    print("\n")
    end

    return p
end


end
