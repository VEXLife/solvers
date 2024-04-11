function [x, feval] = fpi_lsqnonneg(A, b, options)
    %% Use Fixed-point Iteration to solve the least square problem
    % Author: Midden Vexu
    % The problem is min 1/2 * ||Ax - b||_2^2  s.t. x >= 0
    % INPUTS:
    %   A:          m x n matrix
    %   b:          m x 1 vector
    %   options:    struct with fields
    %       x0:                     n x 1 vector, initial guess
    %       MaxIterations:          integer, maximum number of iterations, default 1000
    %       OptimalityTolerance:    double, tolerance for the optimality condition, default 1e-6
    %       verbose:                logical, whether to print the loss at each iteration, default 1
    
    arguments
        A double {mustBeNumeric}
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

    % Similar to KL Divergence.
    % The derivative is 4(A' * (A * x - b)) .* v where x = v .* v
    x = options.x0;
    v = sqrt(x);
    feval_fun = @(x) norm(A * x - b, 2);
    previous_feval = feval_fun(x);
    if options.verbose
        fprintf('Before any iteration, Loss: %f\n', previous_feval);
    end
    
    for i = 1:options.MaxIterations
        numerator = A' * b;
        denominator = A' * (A * x);
        v = v .* ((numerator+denominator) ./ (2*denominator));
        x = v .* v;
        feval = feval_fun(x);
        
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
    end % of function
