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
