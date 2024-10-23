function [x, feval, history, stop_iter] = quadprog_to_lsq_wrapper(A, b, quadprog_solver_func, lambd, varargin)
    % Wrapper for quadprog to solve the nonnegative least squares problem
    % min_x 1/2*||Ax - b||_2^2 + lambd*||x||_1 s.t. x >= 0
    % by converting it into
    % min_x 1/2*(Ax - b)'(Ax - b) = min_x 1/2 * x'A'Ax - b'Ax + b'b / 2
    % and then call the quadprog solver.
    % 
    % Inputs:
    %   A:                      matrix
    %   b:                      vector
    %   quadprog_solver_func:   function handle to the quadprog solver
    %   lambd:                  regularization parameter, default 0
    %                           0 for least squares, >0 for lasso
    %   varargin:               additional arguments for the quadprog solver
    % For more information about the arguments and outputs, refer to the solvers themselves.

    % Set default lambda to 0
    if nargin < 4
        lambd = 0;
    end
        
    % Set up the quadratic programming problem
    Q = A'*A;
    c = lambd-A'*b;
    
    % Solve the quadratic programming problem
    [x, feval, history, stop_iter] = quadprog_solver_func(Q, c, varargin{:});

    % Convert the feval history to the least squares problem
    for i=1:stop_iter+1
        history.feval{i} = history.feval{i} + b'*b / 2;
    end
end