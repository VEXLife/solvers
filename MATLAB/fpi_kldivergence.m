function [x, feval, history, stop_iter] = fpi_kldivergence(A, b, options)
    %% Use Fixed-point Iteration to solve the linear equation problem
    % Author: Midden Vexu
    % The problem is min sum(A * x - b .* log(A * x))  s.t. x >= 0
    % where all entries of matrix A should be non-negative.
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
    
    arguments
        A double {mustBeNumeric, mustBeNonnegative}
        b double {mustBeNumeric, mustBeVector}
        options.x0 double {mustBePositive, mustBeVector}...
            = ones(size(A,2),1)
        options.MaxIterations double {mustBeInteger, mustBePositive}...
            = 1000
        options.OptimalityTolerance double {mustBePositive}...
            = 1e-6
        options.verbose logical...
            {mustBeNumericOrLogical,mustBeMember(options.verbose, [0, 1])}...
            = 1
    end
    
    assert(ismatrix(A), 'A must be a 2D matrix');
    assert(size(A, 1) == numel(b),...
        'The number of rows of A must be equal to the length of b');
    assert(size(A, 2) == numel(options.x0),...
        'The number of columns of A must be equal to the length of x0');
    stop_reason = 'Unexpected stop';

    % Take the original problem as min sum(A * x - b .* log(A * x)) s.t. x >= 0
    % whose derivative is 2 * (A'*vector(1) - A' * (b ./ (A * x)))
    % where vector(1) is a vector of ones;
    % Let the derivative be 0, namely A'*vector(1) = A' * (b ./ (A * x))
    % and multiply x element-wise on both sides, we get A'*vector(1) .* x = A' * (b ./ (A * x)) .* x
    % Then we obtain the iteration step x = (A' * (b ./ (A * x))) ./ (A'*vector(1)) .* x
    % Here we interpolated between the two iterations to ensure the convergence.
    x = options.x0;
    feval_fun = @(x) sum(A * x - b .* log(A * x));
    previous_feval = feval_fun(x);
    history.x = cell(1, options.MaxIterations + 1);
    history.feval = cell(1, options.MaxIterations + 1);
    history.x{1} = x;
    history.feval{1} = previous_feval;
    if options.verbose
        fprintf('Before any iteration, Loss: %f\n', previous_feval);
    end
    
    for i = 1:options.MaxIterations
        numerator = A' * (b ./ (A * x));
        denominator = A' * ones(size(A, 1), 1);
        x = x .* ((numerator+denominator) ./ (2*denominator));
        feval = feval_fun(x);
        history.x{i + 1} = x;
        history.feval{i + 1} = feval;
        
        if abs(feval - previous_feval) < options.OptimalityTolerance
            stop_reason = 'The loss difference is smaller than the optimality tolerance.';
            break
        end
        previous_feval = feval;
        
        if options.verbose
            fprintf('Iteration: %5d, Loss: %f\n', i, feval);
        end
    end % of for loop
    
    if options.verbose
        if i == options.MaxIterations
            stop_reason = 'The maximum number of iterations was reached.';
        end
        fprintf('Stopped after %d iterations because:\n%s\nFinal loss: %f\n', i, stop_reason, feval);
    end
    stop_iter = i;
end % of function
