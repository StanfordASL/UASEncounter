
Power = (x,y)->x^y
Cos = x->cos(x)
Sin = x->sin(x)
Sqrt = x->sqrt(x)

# function heading_hrl(state::EncounterState, D_buffered::Float64, osp::OwnshipParams, isp::IntruderParams)
function heading_hrl(state::EncounterState, action::BoundedHeadingHRL, osp::OwnshipParams, isp::IntruderParams)
    os = state.os
    is = state.is
    xi = is[1]-os[1]
    yi = is[2]-os[2]
    xdoti = isp.v*cos(is[3])
    ydoti = isp.v*sin(is[3])
    D = action.D_buffered
    currpsi = os[3]

    desiredpsi = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])

    # first check distance
    # keep in mind distance must monotonically decrease until toca
    time_bound = action.bound/osp.v
    tau = toca(state)
    if tau < 0.0
        tau = 0.0
    end
    if tau < time_bound
        if dist(state, tau) > D
            return desiredpsi
        end
    else # tau >= time_bound
        if dist(state, time_bound) > D
            return desiredpsi
        end
    end

    # return desiredpsi
    
    # psivals = 0:pi/50:2*pi
    # psivals = currpsi-pi/2:pi/40:currpsi+pi/2
    psivals = linspace(currpsi-pi/2, currpsi+pi/2,40)
    mindists = [mindist(osp.v, psi, xi, yi, xdoti, ydoti) for psi in psivals]

    maxmindist = maximum(mindists)
    if maxmindist < D
        maxinds = findin(mindists, maxmindist)
        if length(maxinds)>1
            maxpsis = psivals[maxinds]
            differences = abs(maxpsis-desiredpsi)
            for i in 1:length(differences)
                while differences[i] > 2*pi differences[i]-=2*pi end
            end
            differences = [min(diff, 2*pi-diff) for diff in differences]
            return maxpsis[indmin(differences)]
        else
            return psivals[maxinds[1]]
        end
    else
        goodinds = find(d->d>=D, mindists)
        goodpsis = psivals[goodinds]
        differences = abs(goodpsis-desiredpsi)
        for i in 1:length(differences)
            while differences[i] > 2*pi differences[i]-=2*pi end
        end
        differences = [min(diff, 2*pi-diff) for diff in differences]
        return goodpsis[indmin(differences)]
    end
end

heading_hrl(state::EncounterState, action::HeadingHRL, osp::OwnshipParams, isp::IntruderParams) = 
    heading_hrl(state, BoundedHeadingHRL(action.D_buffered,Inf), osp, isp)

# vo = 1.0
# psio = 0.0
# xi = 10.0
# yi = 0.0
# xdoti = 0.0
# ydoti = 0.5
# D = 1.0

# vo = 0.1
# psio = pi/2
# xi = -2.0
# yi = 0.0
# xdoti = 1.0
# ydoti = 0.5
# D = 1.0
# 
# f = x->mindist(vo, x, xi, yi, xdoti, ydoti)
# dfdx = x->dmindistdpsi(vo, x, xi, yi, xdoti, ydoti)
