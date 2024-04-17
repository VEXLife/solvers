function [x, feval, history, stop_iter] = pgd_lsqnonneg(A, b, options)
    %% Use Projected Gradient Descent to solve the non-negative least squares problem
    % Author: Midden Vexu
    % Reference: https://angms.science/doc/NMF/nnls_pgd.pdf
    % The problem is min 1/2*||Ax-b||_2^2 s.t. x>=0
    % INPUTS:
    %   A:          m x n matrix
    %   b:          m x 1 vector
    %   options:    struct with fields
    %       x0:                     n x 1 vector, initial guess
    %       MaxIterations:          integer, maximum number of iterations, default 1000
    %       OptimalityTolerance:    double, tolerance for the optimality condition, default 1e-6
    %       LowerBound:             double, can be used to tolerate machine precision, default 0
    %       verbose:                logical, whether to print the loss at each iteration, default 1
    % OUTPUTS:
    %   x:          scalar, solution to the equation
    %   feval:      scalar, final function value at x
    %   history:    struct of two cells, x and feval,
    %               with the first entry being the initial values
    %   stop_iter:  integer, number of iterations before stopping

    arguments
        A double {mustBeNumeric}
        b double {mustBeNumeric, mustBeVector}
        options.x0 double {mustBeNonnegative, mustBeVector}...
            = zeros(size(A,2),1)
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

    assert(ismatrix(A), 'A must be a 2D matrix');
    assert(size(A, 1) == numel(b),...
        'The number of rows of A must be equal to the length of b');
    assert(size(A, 2) == numel(options.x0),...
        'The number of columns of A must be equal to the length of x0');
    stop_reason = 'Unexpected stop';

    theta1 = eye(size(A, 2)) - A' * A / norm(A' * A, "fro");
    theta2 = A' * b / norm(A' * A, "fro");
    x = options.x0;
    y = options.x0;
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
        previous_x = x;
        x = max(options.LowerBound, theta1 * y + theta2);
        y = x + (i - 1) / (i + 2) * (x - previous_x);
        feval = feval_fun(x);
        history.x{i + 1} = x;
        history.feval{i + 1} = feval;
        
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
            history.x{i + 1} = x;
            history.feval{i + 1} = feval;
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

