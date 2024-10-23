function [x, feval, history, stop_iter] = multipupd_quadprognonneg(Q, b, options)
    %% Use Multiplicative Update to solve the non-negative quadratic programming problem
    % Author: Midden Vexu
    % Reference: L. Wu, Y. Yang, and H. Liu, “Nonnegative-lasso and application in index tracking,”
    %            Computational Statistics & Data Analysis, vol. 70, pp. 116–126, Feb. 2014,
    %            doi: 10.1016/j.csda.2013.08.012.
    % The problem is min 1/2 * x' * Q * x + b' * x, s.t. x >= 0
    % INPUTS:
    %   Q:          n x n matrix
    %   b:          n x 1 vector
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
        Q double {mustBeNumeric}
        b double {mustBeNumeric, mustBeVector}
        options.x0 double {mustBeNonzero, mustBeVector}...
            = ones(size(Q,2),1)
        options.MaxIterations double {mustBeInteger, mustBePositive}...
            = 1000
        options.OptimalityTolerance double {mustBePositive}...
            = 1e-6
        options.verbose logical...
            {mustBeNumericOrLogical,mustBeMember(options.verbose, [0, 1])}...
            = 1
    end
    
    n = size(Q, 1);
    assert(ismatrix(Q) && n == size(Q, 2), 'Q must be a 2D square matrix');
    assert(n == numel(b),...
        'The number of rows of Q must be equal to the length of b');
    assert(n == numel(options.x0),...
        'The number of columns of Q must be equal to the length of x0');
    stop_reason = 'Unexpected stop';
    
    Q_pos = max(Q, 0);
    Q_neg = max(-Q, 0);
    x = options.x0;
    feval_fun = @(x) 0.5 * x' * Q * x + b' * x;
    previous_feval = feval_fun(x);
    history.x = cell(1, options.MaxIterations + 1);
    history.feval = cell(1, options.MaxIterations + 1);
    history.x{1} = x;
    history.feval{1} = previous_feval;
    if options.verbose
        fprintf('Before any iteration, Loss: %f\n', previous_feval);
    end
    
    for i = 1:options.MaxIterations
        a = Q_pos * x;
        c = Q_neg * x;
        x = (-b + sqrt(b .^ 2 + 4 * a .* c)) ./ (2 * a) .* x;
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
