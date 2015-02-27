# included into EncounterModel.jl

function dist(state::EncounterState)
    norm(state.os[1:2]-state.is[1:2])
end

function dist(state::EncounterState, t::Float64)
    os = state.os
    is = state.is
    xi = is[1]-os[1]
    yi = is[2]-os[2]
    xdoti = INTRUDER.v*cos(is[3])
    ydoti = INTRUDER.v*sin(is[3])
    psi = os[3]
    return dist(OWNSHIP.v, psi, xi, yi, xdoti, ydoti, t)
end

function toca(state::EncounterState) # time of closest approach
    os = state.os
    is = state.is
    xi = is[1]-os[1]
    yi = is[2]-os[2]
    xdoti = INTRUDER.v*cos(is[3])
    ydoti = INTRUDER.v*sin(is[3])
    psi = os[3]
    return toca(OWNSHIP.v, psi, xi, yi, xdoti, ydoti)
end

function toca(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64)
    # return (-(xdoti*xi) - ydoti*yi + vo*xi*Cos(psio) + vo*yi*Sin(psio))/(Power(vo,2) + Power(xdoti,2) + Power(ydoti,2) - 2*vo*xdoti*Cos(psio) - 2*vo*ydoti*Sin(psio))
    return (-(xdoti*xi) - ydoti*yi + vo*xi*cos(psio) + vo*yi*sin(psio))/(vo^2 + xdoti^2 + ydoti^2 - 2*vo*xdoti*cos(psio) - 2*vo*ydoti*sin(psio))
end

function dist(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64, t::Float64)
    # return Sqrt(Power(t*xdoti + xi - t*vo*Cos(psio),2) + Power(t*ydoti + yi - t*vo*Sin(psio),2))
    return sqrt((t*xdoti + xi - t*vo*cos(psio))^2 + (t*ydoti + yi - t*vo*sin(psio))^2)
end

function mindist(vo::Float64, psio::Float64, xi::Float64, yi::Float64, xdoti::Float64, ydoti::Float64)
    tau = toca(vo, psio, xi, yi, xdoti, ydoti)    
    if tau <= 0.0
        return dist(vo, psio, xi, yi, xdoti, ydoti, 0.0)
    else
        return dist(vo, psio, xi, yi, xdoti, ydoti, tau)
    end
end
function mindist(state::EncounterState)
    tau = toca(state)
    return dist(state, tau)
end
