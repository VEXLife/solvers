function [x, feval] = quadprog_to_lsq_wrapper(A, b, quadprog_solver_func, lambd, varargin)
    % Wrapper for quadprog to solve the nonnegative least squares problem
    % min_x 1/2*||Ax - b||_2^2 + lambd*||x||_1 s.t. x >= 0
    % by converting it into
    % min_x 1/2*(Ax - b)'(Ax - b) = min_x 1/2 * x'A'Ax - b'Ax + b'b / 2
    % and then call the quadprog solver.
    % 
    % Inputs:
    % - A: matrix
    % - b: vector
    % - quadprog_solver_func: function handle to the quadprog solver
    % 
    % Outputs:
    % - x: solution to the least squares problem
    % - feval: struct with information about the evaluation of the solver

    % Set default lambda to 0
    if nargin < 4
        lambd = 0;
    end
        
    % Set up the quadratic programming problem
    Q = A'*A;
    c = lambd-A'*b;
    
    % Solve the quadratic programming problem
    [x, feval] = quadprog_solver_func(Q, c, varargin{:});
end