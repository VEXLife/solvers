%% Test the bisection algorithm
f = @(x) -x+2;
assert(abs(bisection_fsolve(f, 0) - 2) < 1e-6);