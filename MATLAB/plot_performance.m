%% Plot the performance of the solvers
close all;
clear;

% Generate a non-negative linear equation problem
rng(1);
a = randn(30, 3) .^ 2;
x = rand(3, 1);
b = a * x;
[m, n] = size(a);

assert(m == length(b), 'The first dimension of matrix a must match the length of vector b.');

disp("The problem is");
for i = 1:m
    equation = '';
    for j = 1:n - 1
        equation = sprintf('%s%d * x%d + ', equation, a(i, j), j);
    end
    equation = sprintf('%s%d * x%d = b%d', equation, a(i, n), n, i);
    disp(equation);
end
disp("and a possible solution is");
for j = 1:n
    fprintf('x%d = %f\n', j, x(j));
end

% Solve the problem and plot
h=figure;
ax=axes(h);
hold(ax,"on");
[~, ~, history, stop_iter] = quadprog_to_lsq_wrapper(a, b, @pgd_quadprognonneg);
plot(ax,0:stop_iter, cell2mat(history.feval),"DisplayName","Least Squares Projected Gradient Descent");
[~, ~, history, stop_iter] = quadprog_to_lsq_wrapper(a, b, @multipupd_quadprognonneg);
plot(ax,0:stop_iter, cell2mat(history.feval),"DisplayName","Least Squares Multiplicative Update");
[~, ~, history, stop_iter] = gd_kldivergence(a, b);
plot(ax,0:stop_iter, cell2mat(history.feval) + sum(b .* log(b) - b),"DisplayName","KL Divergence Gradient Descent");
[~, ~, history, stop_iter] = fpi_kldivergence(a, b);
plot(ax,0:stop_iter, cell2mat(history.feval) + sum(b .* log(b) - b),"DisplayName","KL Divergence Fixed-point Iteration");
[~, ~, history, stop_iter] = fpi_lsqnonneg(a, b);
plot(ax,0:stop_iter, cell2mat(history.feval),"DisplayName","Least Squares Fixed-point Iteration");
xlim(ax,[0,15]);
title(ax,"Performance of the linear equation solvers");
xlabel(ax,"Iteration");
ylabel(ax,"Loss");
legend(ax);
hold(ax,"off");
saveas(h,"figs/linear_eqn.png");

% Generate a scalar equation problem
a = randn(1, 1);
x = rand(1, 1);
b = a * x + rand(1, 1);

% Solve and plot
h2=figure;
ax2=axes(h2);
hold(ax2,"on");
[~, ~, history, stop_iter] = bisection_fsolve(@(x) a * x - b, 0, GuessRange=[0, 1]);
plot(ax2,0:stop_iter, cell2mat(history.feval),"DisplayName","Bisection");
title(ax2,"Performance of the scalar equation solvers");
xlabel(ax2,"Iteration");
ylabel(ax2,"Function value");
legend(ax2);
hold(ax2,"off");
saveas(h2,"figs/scalar_eqn.png");