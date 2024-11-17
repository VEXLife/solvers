function [x, feval, history, stop_iter] = sor_lsqr(A, b, options)
    %% Use SOR(Successive Over-Relaxation) Fixed-Point Iteration method to solve least squares problem
    % Author: Midden Vexu
    % The problem is min ||Ax - b||_2^2
    % where A is a matrix and b is a vector.
    % INPUTS:
    %   A:          m x n matrix
    %   b:          m x 1 vector
    %   options:    struct with fields
    %       x0:                     n x 1 vector, initial guess
    %       MaxIterations:          integer, maximum number of iterations, default 1000
    %       OptimalityTolerance:    double, tolerance for the optimality condition, default 1e-6
    %       verbose:                logical, whether to print the loss at each iteration, default 1
    % OUTPUTS:
    %   x:          scalar, solution to the equation
    %   feval:      scalar, final function value at x
    %   history:    struct of two cells, x and feval,
    %               with the first entry being the initial values
    %   stop_iter:  integer, number of iterations before stopping
    
    arguments (Input)
        A double {mustBeNumeric}
        b double {mustBeNumeric, mustBeVector}
        options.x0 double {mustBeVector}...
            = ones(size(A,2),1)
        options.MaxIterations double {mustBeInteger, mustBePositive}...
            = 1000
        options.OptimalityTolerance double {mustBePositive}...
            = 1e-6
        options.dampingCoeff double...
            {mustBeNonnegative, mustBeLessThanOrEqual(options.dampingCoeff, 2)}...
            = 0.5
        options.verbose logical...
            {mustBeNumericOrLogical,mustBeMember(options.verbose, [0, 1])}...
            = 1
    end

    arguments (Output)
        x double
        feval double
        history struct
        stop_iter double
    end
    
    assert(ismatrix(A), 'A must be a 2D matrix');
    assert(size(A, 1) == numel(b),...
        'The number of rows of A must be equal to the length of b');
    assert(size(A, 2) == numel(options.x0),...
        'The number of columns of A must be equal to the length of x0');
    stop_reason = 'Unexpected stop';

    x = options.x0;
    d = diag(diag(A));
    l = tril(A, -1);
    u = triu(A, 1);
    feval_fun = @(x) norm(A * x - b, 2);
    previous_feval = feval_fun(x);
    history.x = cell(1, options.MaxIterations + 1);
    history.feval = cell(1, options.MaxIterations + 1);
    history.x{1} = x;
    history.feval{1} = previous_feval;
    if options.verbose
        fprintf('Before any iteration, Loss: %f\n', previous_feval);
    end
    
    for i = 1:options.MaxIterations
        x = (d + options.dampingCoeff * l) \ ((1 - options.dampingCoeff) * d - options.dampingCoeff * u) * x + options.dampingCoeff * ((d + options.dampingCoeff * l) \ b);

        feval = feval_fun(x);
        history.x{i + 1} = x;
        history.feval{i + 1} = feval;
        
        if options.verbose
            fprintf('Iteration: %5d, Loss: %f\n', i, feval);
        end
        
        if abs(feval - previous_feval) < options.OptimalityTolerance
            stop_reason = 'The loss difference is smaller than the optimality tolerance.';
            break
        end

        previous_feval = feval;
    end % of for loop
    
    if options.verbose
        if i == options.MaxIterations
            stop_reason = 'The maximum number of iterations was reached.';
        end
        fprintf('Stopped after %d iterations because:\n%s\nFinal loss: %f\n', i, stop_reason, feval);
    end
    stop_iter = i;
end % of function
