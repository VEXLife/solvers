function [x, feval, history, stop_iter] = gd_kldivergence(A, b, options)
    %% Use Gradient Descent to solve the KL Divergence optimization problem
    % Author: Midden Vexu
    % Reference: A.-A. Lu, Y. Chen, and X. Gao,
    %            “2D Beam Domain Statistical CSI Estimation for Massive MIMO Uplink,”
    %            IEEE Transactions on Wireless Communications, pp. 1–1, 2023, doi: 10.1109/TWC.2023.3281841.
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
    %       StepSize:               double, initial step size, default 1
    %       StepSizeMinimum:        double, stops if the step size is smaller than this value, default 1e-4
    %       StepSizeDiscount:       double, discount factor for the step size, default 5e-1
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
        options.StepSize double {mustBePositive}...
            = 1
        options.StepSizeMinimum double {mustBePositive}...
            = 1e-4
        options.StepSizeDiscount double ...
            {mustBePositive, mustBeLessThanOrEqual(options.StepSizeDiscount, 1)}...
            = 5e-1
    end
    
    assert(ismatrix(A), 'A must be a 2D matrix');
    assert(size(A, 1) == numel(b),...
        'The number of rows of A must be equal to the length of b');
    assert(size(A, 2) == numel(options.x0),...
        'The number of columns of A must be equal to the length of x0');
    stop_reason = 'Unexpected stop';

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
        previous_v = v;
        gradient = 2 * A' * ones(size(A, 1), 1) .* v - 2 * A' * (b ./ (A * x)) .* v;
        v = v - options.StepSize * gradient;
        x = v .* v;
        feval = feval_fun(x);
        history.x{i + 1} = x;
        history.feval{i + 1} = feval;
        
        if abs(feval - previous_feval) < options.OptimalityTolerance
            stop_reason = 'The loss difference is smaller than the optimality tolerance.';
            break
        end
        if feval > previous_feval
            options.StepSize = options.StepSize * options.StepSizeDiscount;
            v = previous_v;
            x = v .* v;
            feval = feval_fun(x);
            history.x{i + 1} = x;
            history.feval{i + 1} = feval;

            if options.verbose
                fprintf('Returned to the previous point as the loss increased. New step size: %f\n', options.StepSize);
            end

            if options.StepSize < options.StepSizeMinimum
                stop_reason = 'The step size is smaller than the minimum step size.';
                break
            end
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
