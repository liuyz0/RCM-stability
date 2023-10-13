# dynamics function

function ResCon!(du,u,para,t)
    N_s,N_r,G,C,gamma,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ) + 1e-7*ones(N_s) 
    du[N_s+1:N_r+N_s] = gamma - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

function ResConLin!(du,u,para,t)
    N_s,N_r,G,C,l,kappa,δ = para
    du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ) + 1e-7*ones(N_s) 
    du[N_s+1:N_r+N_s] = l.*(kappa - u[N_s+1:N_r+N_s]) - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
end

function ResConLog!(du,u,para,t)
    N_s,N_r,G,C,g,K,δ = para
    du[1:N_s] = u[1:N_s] .* (G*u[N_s+1:N_r+N_s] - δ) + 1e-7*ones(N_s)
    du[N_s+1:N_r+N_s] = u[N_s+1:N_r+N_s] .* (g.*(K - u[N_s+1:N_r+N_s]) - C'*u[1:N_s]) + 1e-7*ones(N_r)
end

# solving


function solRCM(Ss,Rs,para,RCM,disturb=1.99,tEnd=5000.0,MIN_POP = 1e-9)
    Ns, Nr = para
    tspan = (0.0, tEnd)
    alg = AutoVern7(Rodas4())
    
    # checks if species fall below extinction threshold
    condition(u, t, integrator) = any(u .< MIN_POP)
    # sets species below extinction threshold to 0
    function affect!(integrator)
        integrator.u[integrator.u.<MIN_POP] .= 0.0
    end
    cb = DiscreteCallback(condition, affect!)
    
    # sample an initial condition
    u0 = zeros(Ns+Nr)
    u0[1:Ns] = Ss .* (1 .+ disturb*(rand(Ns) .- 0.5))
    u0[Ns+1:end] = Rs .* (1 .+ disturb*(rand(Nr) .- 0.5))
    iter = 1

    #initialize output
    output = zeros(3)

    stop = 0
    while stop == 0
        prob = ODEProblem(RCM,u0,tspan,para)
        #sol = solve(prob, saveat = 10.0, alg, isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
        sol = solve(prob, saveat = 10.0, alg, callback = cb)
        meanS = mean(sol.u[end-100:end])[1:Ns]
        index = findall(meanS .> 1e-5)
        flagS = mean(abs.((sol.u[end][index] - meanS[index])./meanS[index]))
        if iter == 15 || flagS <= 1e-2
            stop = 1
            output[1] = length(index)/Ns # fraction of survial
        end
        iter = iter + 1
        u0 = sol.u[end]
        #u0[u0 .< 1.0e-9] .= 1.0e-9
    end

    fluc = 0.0
    if iter == 16
        fluc = 1.0
    end 

    if fluc == 1.0
        output[2] = 1.0 # flucturate
    else
        output[3] = 1.0 # stable
    end

    return output, u0
end