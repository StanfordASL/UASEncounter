
Power = (x,y)->x^y
Cos = x->cos(x)
Sin = x->sin(x)
Sqrt = x->sqrt(x)

# function dmindistdpsi(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64)
#     deriv = (vo*(-(xi*ydoti) + xdoti*yi - vo*yi*Cos(psio) + vo*xi*Sin(psio))*(vo - xdoti*Cos(psio) - ydoti*Sin(psio))*(-(xdoti*xi) - ydoti*yi + vo*xi*Cos(psio) + vo*yi*Sin(psio)))/ (Sqrt(Power(xi*ydoti - xdoti*yi + vo*yi*Cos(psio) - vo*xi*Sin(psio),2)/(Power(vo,2) + Power(xdoti,2) + Power(ydoti,2) - 2*vo*xdoti*Cos(psio) - 2*vo*ydoti*Sin(psio)))* Power(Power(vo,2) + Power(xdoti,2) + Power(ydoti,2) - 2*vo*xdoti*Cos(psio) - 2*vo*ydoti*Sin(psio),2))
#     if deriv != deriv # check for NaN
#         return dmindistdpsi(vo, psio+0.0001, xi, yi, xdoti, ydoti)
#     end
#     return deriv
# end

# function dist(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64, t::Float64)
#     return Sqrt(Power(t*xdoti + xi - t*vo*Cos(psio),2) + Power(t*ydoti + yi - t*vo*Sin(psio),2))
# end
# 
# function mindist(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64)
#     toca = (-(xdoti*xi) - ydoti*yi + vo*xi*Cos(psio) + vo*yi*Sin(psio))/(Power(vo,2) + Power(xdoti,2) + Power(ydoti,2) - 2*vo*xdoti*Cos(psio) - 2*vo*ydoti*Sin(psio))
#     if toca <= 0.0
#         return dist(vo, psio, xi, yi, xdoti, ydoti, 0.0)
#     else
#         return dist(vo, psio, xi, yi, xdoti, ydoti, toca)
#     end
# end

function heading_hrl(state::EncounterState, D_buffered::Float64, osp::OwnshipParams, isp::IntruderParams)
    os = state.os
    is = state.is
    xi = is[1]-os[1]
    yi = is[2]-os[2]
    xdoti = isp.v*cos(is[3])
    ydoti = isp.v*sin(is[3])
    D = D_buffered
    currpsi = os[3]

    desiredpsi = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])

    # return desiredpsi
    
    # psivals = 0:pi/50:2*pi
    psivals = currpsi-pi/2:pi/20:currpsi+pi/2
    mindists = [mindist(osp.v, psi, xi, yi, xdoti, ydoti) for psi in psivals]

    maxmindist = maximum(mindists)
    if maxmindist < D
        # @show maxmindist
        # @show mindists
        # @show [toca(osp.v, psi, xi, yi, xdoti, ydoti) for psi in psivals]
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
