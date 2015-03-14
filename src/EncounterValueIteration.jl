
module EncounterValueIteration

using EncounterModel: IntruderParams, OwnshipParams, SimParams, EncounterState, EncounterAction, RewardModel, ownship_dynamics, encounter_dynamics, next_state_from_pd, post_decision_state, reward, SIM
using EncounterFeatures: FeatureBlock, f_focused_intruder_grid, evaluate
using EncounterSimulation: LinearQValuePolicy, LinearPostDecisionPolicy
using GridInterpolations
# import SVDSHack
import HDF5, JLD
import Dates

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

function snap_os_to_goal_grid(state::EncounterState, goal_grid)
    # snap os to nearest goal point
    os = state.os

    d_goal = min(norm(os[1:2]-SIM.goal_location)-SIM.goal_radius, maximum(goal_grid.cutPoints[1]))

    if d_goal > 0.0 && d_goal <= maximum(goal_grid.cutPoints[1])
        heading = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])
        if heading <= 0.0 heading += 2*pi end # heading now 0 to 2*pi
        bearing = heading-os[3]
        while bearing < 0.0 bearing += 2*pi end
        while bearing >= 2*pi bearing -= 2*pi end

        inds, weights = interpolants(goal_grid, [d_goal, bearing])

        (dgnew,bgnew) = ind2x(goal_grid, inds[indmax(weights)])
        if dgnew < 1e-5
            dgnew += 1e-5
        end

        state.os[1:2] = (os[1:2]-SIM.goal_location)*(dgnew+SIM.goal_radius)/(d_goal+SIM.goal_radius)+SIM.goal_location
        state.os[3] = heading-bgnew
    end
    
    return state
end

function snap_is_to_intruder_grid(state::EncounterState, intruder_grid)
    # snap is so that os is on nearest
    os = state.os
    is = state.is

    d = norm(os[1:2]-is[1:2])-SIM.legal_D

    # bearing to ownship from intruder's perspective
    bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    while bearing >= pi bearing -= 2*pi end
    while bearing < -pi bearing += 2*pi end
    # relative heading of ownship compared to intruder
    heading = os[3] - is[3]
    while heading >= 2*pi heading -= 2*pi end
    while heading < 0.0 heading += 2*pi end

    if bearing > pi/2 || bearing < -pi/2 || d <= 0.0 || d > maximum(intruder_grid.cutPoints[1])
        return state
    end

    inds, weights = interpolants(intruder_grid, [d, bearing, heading])

    # @show ind2x(intruder_grid, inds[indmax(weights)])
    (dnew,bnew,hnew) = ind2x(intruder_grid, inds[indmax(weights)])
    if dnew == 0.0
        dnew += 1e-5
    end

    state.is[3] = os[3] - hnew
    state.is[1:2] = os[1:2] - [(dnew+SIM.legal_D)*cos(is[3]+bnew), (dnew+SIM.legal_D)*sin(is[3]+bnew)]

    return state
end

function gen_state_snap_to_grid(rng::AbstractRNG, intruder_grid, goal_grid)
    state=gen_state(rng)
    return snap_is_to_intruder_grid(snap_os_to_goal_grid(state, goal_grid),intruder_grid)
end

# HACK
function gen_ic_batch_for_grid(rng, intruder_grid, goal_grid)
    ics = Array(EncounterState, length(intruder_grid)+length(goal_grid))
    for i in 1:length(intruder_grid)
        ix = 1000.0*rand(rng)
        iy = 2000.0*(rand(rng)-0.5)
        ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))
        is = [ix, iy, ihead]
        (dnew,bnew,hnew) = ind2x(intruder_grid,i)
        if dnew <= 0.0 dnew+=1e-5 end
        if bnew > pi/2-1e-5 bnew-=1e-5 end
        if bnew < -pi/2+1e-5 bnew+=1e-5 end
        if hnew < 1e-5 hnew+=1e-5 end
        if hnew > 2*pi-1e-5 hnew-=1e-5 end
        os = [is[1]+(dnew+SIM.legal_D)*cos(is[3]+bnew),
              is[2]+(dnew+SIM.legal_D)*sin(is[3]+bnew),
              is[3]+hnew]
        ics[i] = EncounterState(os, is, false)
        feat = f_focused_intruder_grid(ics[i],intruder_grid)
        if length(find(feat))==0
            @show (dnew,bnew,hnew)
        end
    end
    for i in 1:length(goal_grid)
        (dnew,bnew) = ind2x(goal_grid,i)
        if dnew <= 0.0 dnew+=1e-5 end
        if bnew > pi/2-1e-5 bnew-=1e-5 end
        if bnew < -pi/2+1e-5 bnew+=1e-5 end
        head = 2*pi*rand(rng)
        oxy = SIM.goal_location-(dnew+SIM.goal_radius)*[cos(head+bnew), sin(head+bnew)]
        ix = 1000.0*rand(rng)
        iy = 2000.0*(rand(rng)-0.5)
        ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))
        is = [ix, iy, ihead]
        state = EncounterState([oxy[1],oxy[2],head],is,false,false)
        state = snap_is_to_intruder_grid(state, intruder_grid)
        ics[length(intruder_grid)+i] = state
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
                     intruder_grid::AbstractGrid,
                     goal_grid::AbstractGrid;
                     post_decision=false,
                     parallel=true,
                     num_short=30,
                     num_long=1)

    rng0 = MersenneTwister(0)

    theta = zeros(length(phi))
    snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid, goal_grid)

    for i in 1:
        tic()
        sims_per_policy = 10000
        # println("starting value iteration $i ($sims_per_policy simulations)")
        ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid,goal_grid)
        theta_new = iterate(phi, theta, rm, actions, sims_per_policy,
                            rng_seed_offset=2048*i,
                            state_gen=snap_generator,
                            parallel=true,
                            ic_batch=ic_batch,
                            output_prefix="\r[$i ($sims_per_policy)]",
                            output_suffix="",
                            parallel=parallel)
        theta = theta_new
        toc()
    end

    for j in 1:num_long
        tic()
        sims_per_policy = 50000
        # println("starting final value iteration ($sims_per_policy simulations)")
        ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid,goal_grid)
        theta_new = iterate(phi, theta, rm, actions, sims_per_policy,
                            rng_seed_offset=2011*j,
                            state_gen=snap_generator,
                            parallel=true,
                            ic_batch=ic_batch,
                            output_prefix="\r[final ($sims_per_policy)]",
                            output_suffix=""
                            parallel=parallel)
        theta = theta_new
        toc()
    end

    ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid,goal_grid)
    if post_decision
        return extract_pd_policy(phi, theta, actions, 50000,
                            ic_batch=ic_batch,
                            state_gen=snap_generator)
    else
        return extract_policy(phi, theta, rm, actions, 50000,
                            ic_batch=ic_batch,
                            state_gen=snap_generator,
                            parallel=parallel)
    end
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
            push!(refs, @spawn run_sims(new_phi, phi, theta, rm, actions, sims_per_spawn, num_EV, rng_seed_offset+num_sims*i, state_gen=state_gen, ic_batch=ic_batch_part))
        else
            push!(refs, run_sims(new_phi, phi, theta, rm, actions, sims_per_spawn, num_EV, rng_seed_offset+num_sims*i, state_gen=state_gen, ic_batch=ic_batch_part))
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

    try
        new_theta = pinv(Phi)*v
    catch e
        @show rank(Phi)
        println(e)
        JLD.@save "pinv_crash_$(Dates.format(Dates.now(),"u-d_HHMM")).jld" phi theta rm actions num_sims new_phi num_EV rng_seed_offset sims_per_spawn convert_to_sparse parallel ic_batch
        new_theta = theta
    end
    if length(new_theta) <= 100
        @show new_theta
    end
    # toc()

    return new_theta
end

function extract_pd_policy(phi::FeatureBlock,
                            theta::AbstractVector{Float64},
                            actions::Vector{EncounterAction},
                            num_sims::Int;
                            new_phi=nothing,
                            num_EV::Int=20,
                            state_gen::Function=gen_state, 
                            ic_batch::Vector{EncounterState}=EncounterState[])

    rng = MersenneTwister(0)
    if new_phi==nothing
        new_phi = phi
    end
    new_theta = Array(Float64, length(new_phi))

    println("running sims...")

    Phi = Array(Float64, num_sims, length(new_theta))
    v = Array(Float64, num_sims)
    for n = 1:num_sims
        if n<=length(ic_batch)
            pd = ic_batch[n]
        else
            pd = state_gen(rng)
        end
        Phi[n,:] = evaluate(new_phi,pd)
        vn = 0
        for l = 1:num_EV
            sp = next_state_from_pd(pd, rng)
            vn += sum(evaluate(phi, sp)'*theta)
        end
        v[n]= vn/num_EV
    end
    
    println("inverting...")
    new_theta=pinv(Phi)*v

    return LinearPostDecisionPolicy(new_phi, actions, new_theta)

end

function extract_policy(phi::FeatureBlock,
                        theta::AbstractVector{Float64},
                        rm::RewardModel,
                        actions::Vector{EncounterAction},
                        num_sims::Int;
                        new_phi=nothing,
                        num_EV::Int=20,
                        state_gen::Function=gen_state, 
                        ic_batch::Vector{EncounterState}=EncounterState[],
                        parallel=true)

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
                              output_suffix="",
                              parallel=parallel)
    print("\n")
    end

    return p
end


#     state=gen_state(rng)
# 
#     is = state.is
#     os = state.os
# 
#     d = norm(os[1:2]-is[1:2])-SIM.legal_D
# 
#     # bearing to ownship from intruder's perspective
#     bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
#     while bearing >= pi bearing -= 2*pi end
#     while bearing < -pi bearing += 2*pi end
#     # relative heading of ownship compared to intruder
#     heading = os[3] - is[3]
#     while heading >= 2*pi heading -= 2*pi end
#     while heading < 0.0 heading += 2*pi end
# 
#     if bearing > pi/2 || bearing < -pi/2 || d <= 0.0 || d > maximum(intruder_grid.cutPoints[1])
#         return state
#     end
# 
#     inds, weights = interpolants(intruder_grid, [d, bearing, heading])
# 
#     # @show ind2x(intruder_grid, inds[indmax(weights)])
#     (dnew,bnew,hnew) = ind2x(intruder_grid, inds[indmax(weights)])
#     if dnew == 0.0
#         dnew += 1e-5
#     end
# 
#     state.os[1:2] = is[1:2] + [(dnew+SIM.legal_D)*cos(is[3]+bnew), (dnew+SIM.legal_D)*sin(is[3]+bnew)]
#     state.os[3] = hnew+is[3]
# 
#     # DEBUG
#     # is = state.is
#     # os = state.os
# 
#     # d = norm(os[1:2]-is[1:2])-SIM.legal_D
# 
#     # # bearing to ownship from intruder's perspective
#     # bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
#     # while bearing >= pi bearing -= 2*pi end
#     # while bearing < -pi bearing += 2*pi end
#     # # relative heading of ownship compared to intruder
#     # heading = os[3] - is[3]
#     # while heading >= 2*pi heading -= 2*pi end
#     # while heading < 0.0 heading += 2*pi end
# 
#     # @show (dnew,d)
#     # @show (bnew,bearing)
#     # @show (hnew,heading)
# 
#     return state
# end


# HACK HACK HACK
# this is such a pain!!!
# I hate this
# function gen_state_snap_to_grid(rng::AbstractRNG, intruder_grid, goal_grid)
#     state=gen_state(rng)
# 
#     # snap os to nearest goal point
#     os = state.os
# 
#     d_goal = min(norm(os[1:2]-SIM.goal_location)-SIM.goal_radius, maximum(goal_grid.cutPoints[1]))
# 
#     if d_goal > 0.0 && d_goal <= maximum(goal_grid.cutPoints[1])
#         heading = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])
#         if heading <= 0.0 heading += 2*pi end # heading now 0 to 2*pi
#         bearing = heading-os[3]
#         while bearing < 0.0 bearing += 2*pi end
#         while bearing >= 2*pi bearing -= 2*pi end
# 
#         inds, weights = interpolants(goal_grid, [d_goal, bearing])
# 
#         (dgnew,bgnew) = ind2x(goal_grid, inds[indmax(weights)])
#         if dgnew < 1e-5
#             dgnew += 1e-5
#         end
# 
#         state.os[1:2] = (os[1:2]-SIM.goal_location)*(dgnew+SIM.goal_radius)/(d_goal+SIM.goal_radius)+SIM.goal_location
#         state.os[3] = heading-bgnew
#     end
# 
#     # snap is so that os is on nearest
#     os = state.os
#     is = state.is
# 
#     d = norm(os[1:2]-is[1:2])-SIM.legal_D
# 
#     # bearing to ownship from intruder's perspective
#     bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
#     while bearing >= pi bearing -= 2*pi end
#     while bearing < -pi bearing += 2*pi end
#     # relative heading of ownship compared to intruder
#     heading = os[3] - is[3]
#     while heading >= 2*pi heading -= 2*pi end
#     while heading < 0.0 heading += 2*pi end
# 
#     if bearing > pi/2 || bearing < -pi/2 || d <= 0.0 || d > maximum(intruder_grid.cutPoints[1])
#         return state
#     end
# 
#     inds, weights = interpolants(intruder_grid, [d, bearing, heading])
# 
#     # @show ind2x(intruder_grid, inds[indmax(weights)])
#     (dnew,bnew,hnew) = ind2x(intruder_grid, inds[indmax(weights)])
#     if dnew == 0.0
#         dnew += 1e-5
#     end
# 
#     state.is[3] = os[3] - hnew
#     state.is[1:2] = os[1:2] - [(dnew+SIM.legal_D)*cos(is[3]+bnew), (dnew+SIM.legal_D)*sin(is[3]+bnew)]
# 
#     # state.os[1:2] = is[1:2] + [(dnew+SIM.legal_D)*cos(is[3]+bnew), (dnew+SIM.legal_D)*sin(is[3]+bnew)]
#     # state.os[3] = hnew+is[3]
# 
#     # DEBUG
#     # is = state.is
#     # os = state.os
# 
#     # d = norm(os[1:2]-is[1:2])-SIM.legal_D
# 
#     # # bearing to ownship from intruder's perspective
#     # bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
#     # while bearing >= pi bearing -= 2*pi end
#     # while bearing < -pi bearing += 2*pi end
#     # # relative heading of ownship compared to intruder
#     # heading = os[3] - is[3]
#     # while heading >= 2*pi heading -= 2*pi end
#     # while heading < 0.0 heading += 2*pi end
# 
#     # @show (dnew,d)
#     # @show (bnew,bearing)
#     # @show (hnew,heading)
# 
#     return state
# end


end
