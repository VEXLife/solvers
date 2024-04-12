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
        warning('MATLAB:FixedPointIterationMayDiverge',...
            "Fixed-point iteration algorithm may diverge with a bad initial guess, you'd better switch to more robust algorithms."); 
    stop_reason = 'Unexpected stop';

    % Take the original problem as min sum(A * x - b .* log(A * x)) s.t. x = v .* v
    % and substituting x = v .* v, we get an unconstrained problem
    % min sum(A(v .* v) - b .* log(A(v .* v)))
    % whose derivative is 2 * ((A'*vector(1)) .* v - (A' * (b ./ (A * v))) .* v)
    % where vector(1) is a vector of ones;
    % Let the derivative be 0, namely (A'*vector(1)) .* v = (A' * (b ./ (A * v))) .* v
    % and we obtain the iteration step v = (A' * (b ./ (A * v))) ./ (A'*vector(1)) .* v
    % Here we interpolated between the two iterations to mitigate divergence
    x = options.x0;
    v = sqrt(x);
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
        v = v .* ((numerator+denominator) ./ (2*denominator));
        x = v .* v;
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
