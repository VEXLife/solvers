%% Test the bisection algorithm
f = @(x) -x+2;
assert(abs(bisection_eqnsolve(f) - 2) < 1e-6);