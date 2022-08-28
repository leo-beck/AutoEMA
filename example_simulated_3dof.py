from AutoEMA import AutoEMA as ae

# Load data
frf, f = ae.load_example()

# Init model
model = ae.OptModel(frf=frf, f_axis=f)
# Optimize model
model.optimize(n_init=2, n_iter=2)  # Do more iterations on real data

# Plot stability diagram
model.plot_stability_diagram()

# Print information
print(model)

# Get results
reconstructed_frf, freq_axis, nat_freqs, damp_ratios, mode_shapes = model.get_results()
FRAC = model.get_frac()

# Get optimized parameters and use them without iterative optimization
p = model.params
bmodel = ae.BaseModel(frf=frf, f_axis=f, params=p)
bmodel.run()

# Optimizing can be continued without loosing information
model.optimize(n_init=0, n_iter=1)

# Store model
ae.save_model(model, "my_model")
m = ae.load_model("my_model")
