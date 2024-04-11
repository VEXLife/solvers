function [x, feval] = pgd_quadprognonneg(Q, b, options)
    %% Use Projected Gradient Descent to solve the non-negative quadratic programming problem
    % Author: Midden Vexu
    % Reference: https://angms.science/doc/NMF/nnls_pgd.pdf
    % The problem is min 1/2 * x' * Q * x + b' * x, s.t. x >= 0
    % INPUTS:
    %   Q:          n x n matrix
    %   b:          n x 1 vector
    %   options:    struct with fields
    %       x0:                     n x 1 vector, initial guess
    %       MaxIterations:          integer, maximum number of iterations, default 1000
    %       OptimalityTolerance:    double, tolerance for the optimality condition, default 1e-6
    %       LowerBound:             double, can be used to tolerate machine precision, default 0
    %       verbose:                logical, whether to print the loss at each iteration, default 1
    
    arguments
        Q double {mustBeNumeric}
        b double {mustBeNumeric, mustBeVector}
        options.x0 double {mustBeNonnegative, mustBeVector}...
            = zeros(size(Q,2),1)
        options.MaxIterations double {mustBeInteger, mustBePositive}...
            = 1000
        options.OptimalityTolerance double {mustBePositive}...
            = 1e-6
        options.LowerBound double {mustBeNonnegative, mustBeFinite, mustBeScalarOrEmpty}...
            = 0
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
    
    theta1 = eye(size(Q, 2)) - Q / norm(Q, "fro");
    theta2 = -b / norm(Q, "fro");
    x = options.x0;
    y = options.x0;
    feval_fun = @(x) 0.5 * x' * Q * x + b' * x;
    previous_feval = feval_fun(x);
    if options.verbose
        fprintf('Before any iteration, Loss: %f\n', previous_feval);
    end
    
    for i = 1:options.MaxIterations
        previous_x = x;
        x = max(options.LowerBound, theta1 * y + theta2);
        y = x + (i - 1) / (i + 2) * (x - previous_x);
        feval = feval_fun(x);
        
        if abs(feval - previous_feval) < options.OptimalityTolerance
            stop_reason = 'The loss difference is smaller than the optimality tolerance.';
            break
        end
        if feval > previous_feval
            if options.verbose
                fprintf('Redo a simple gradient descent step as the loss increases\n');
            end
            x = max(options.LowerBound, theta1 * previous_x + theta2);
            y = x; % Restart y from x
            feval = feval_fun(x);
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
    
    