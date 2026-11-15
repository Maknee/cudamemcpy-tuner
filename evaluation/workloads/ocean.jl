# Oceananigans 0.117.1 examples/ocean_wind_mixing_and_convection.jl, run as published except lines marked
# "# changed" / "# added". Plotting (CairoMakie) is removed: in the example it runs after the simulation.
# ENV: OCEAN_N (Nx = Ny, published 128; Nz = N/2), OCEAN_OUT = full (whole 3D fields) | slice (published xz slice),
#      OCEAN_STOP_MIN (published 120), OCEAN_DIR (output directory).
# Copies: each save moves every output field GPU -> host (pageable DtoH, one field = N*N*N/2*8 bytes, 64 MiB at
# N = 256) before JLD2 writes it. Measured with `cmt trace` on an A100, N = 256 full: 8.5 GiB DtoH in 20 saves.
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using Random
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
Random.seed!(1969) # for reproducible results
Nx = Ny = parse(Int, get(ENV, "OCEAN_N", "128"))  # changed
Nz = Nx ÷ 2  # changed (published: 64 with Nx = 128)
Lx = Ly = 128meters
Lz = 64meters
refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_interfaces(k) = Lz * (ζ₀(k) * Σ(k) - 1)
grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_interfaces)
ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
equation_of_state = TEOS10EquationOfState(reference_density=ρₒ)
buoyancy = SeawaterBuoyancy(; equation_of_state)
Q = 200   # W m⁻², surface _heat_ flux
cᴾ = 3991 # J K⁻¹ kg⁻¹, typical heat capacity for seawater
Jᵀ = Q / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux
dTdz = 0.01 # K m⁻¹
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ),
                                bottom = GradientBoundaryCondition(dTdz))
u₁₀ = 10  # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2e-3 # dimensionless drag coefficient
ρₐ = 1.2  # kg m⁻³, approximate average density of air at sea-level
τx = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@inline Jˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
evaporation_rate = 1e-3 / hour # m s⁻¹
evaporation_bc = FluxBoundaryCondition(Jˢ, field_dependencies=:S, parameters=evaporation_rate)
S_bcs = FieldBoundaryConditions(top=evaporation_bc)
model = NonhydrostaticModel(grid; buoyancy,
                            advection = WENO(order=7),
                            tracers = (:T, :S),
                            coriolis = FPlane(f=1e-4),
                            closure = DynamicSmagorinsky(),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))
Ξ(z) = randn() * z / Lz * (1 + z / Lz)
Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * Lz * 2e-6 * Ξ(z)
uᵢ(x, y, z) = sqrt(abs(τx)) * 1e-3 * Ξ(z)
Random.seed!(1969)  # added: same initial noise in every run, whatever the setup drew from the RNG
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)
simulation = Simulation(model, Δt=10, stop_time=parse(Float64, get(ENV, "OCEAN_STOP_MIN", "120"))minutes)  # changed
conjure_time_step_wizard!(simulation, cfl=0.7)
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))
add_callback!(simulation, progress_message, IterationInterval(200))
eddy_viscosity = (; νₑ = model.closure_fields.νₑ)
filename = joinpath(get(ENV, "OCEAN_DIR", "."), "ocean_wind_mixing_and_convection.jld2")  # changed: directory
full = get(ENV, "OCEAN_OUT", "full") == "full"  # added
simulation.output_writers[:slices] =
    JLD2Writer(model, merge(model.velocities, model.tracers, eddy_viscosity);
               filename,
               indices = full ? (:, :, :) : (:, Ny ÷ 2, :),  # changed: "full" saves whole 3D fields
               schedule = TimeInterval(1minute),
               overwrite_files = true)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
# added: a 2-iteration run first compiles the time step and output kernels (Julia JIT), then the timed run continues
simulation.stop_iteration = 2; run!(simulation); simulation.stop_iteration = Inf
CUDA.synchronize(); t0 = time()  # added
run!(simulation)
CUDA.synchronize(); dt = time() - t0  # added
@printf("{\"metric\":\"simulation_time\",\"value\":%.3f,\"unit\":\"s\",\"N\":%d,\"iterations\":%d,\"output\":\"%s\",\"file_bytes\":%d}\n",
        dt, Nx, iteration(simulation), full ? "full" : "slice", filesize(filename))  # added
