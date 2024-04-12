function [x, feval, history, stop_iter] = bisection_fsolve(f, x0, options)
    %% Use the bisection method to solve a scalar equation f(x) = 0
    % Author: Midden Vexu
    % **WARNING**: This function does not calculate the derivative of the function,
    % it simply assumes that the function is monotonic inside the guess range.
    % When f(x0) > 0, the algorithm decreases x0;
    % When f(x0) < 0, the algorithm increases x0.
    % INPUTS:
    %   f:          function handle
    %   x0:         scalar, initial guess
    %   options:    struct of option fields
    %       OptimalityTolerance:    double, default 1e-6
    %       MaxIterations:          integer, default 100
    %       GuessRange:             1x2 vector, default [0, 1e6]
    %       verbose:                logical, default 1
    % OUTPUTS:
    %   x:          scalar, solution to the equation
    %   feval:      scalar, final function value at x

    arguments
        f function_handle
        x0 double {mustBeReal}
        options.OptimalityTolerance double {mustBePositive} = 1e-6
        options.MaxIterations int16 {mustBePositive} = 100
        options.GuessRange (1, 2) double {mustBeReal} = [0, 1e6]
        options.verbose logical...
            {mustBeNumericOrLogical,mustBeMember(options.verbose, [0, 1])}...
            = 1
    end
    
    % Initialize the bisection method
    x_max = options.GuessRange(2);
    x_min = options.GuessRange(1);
    assert(f(x_min) * f(x_max) < 0, 'The function does not change sign in the guess range, meaning there may be no solutions in the range.');
    x = x0;
    stop_reason = 'Unexpected stop';
    right_sign = sign(f(x_max));

    feval = f(x);
    history.x = cell(1, options.MaxIterations + 1);
    history.feval = cell(1, options.MaxIterations + 1);
    history.x{1} = x;
    history.feval{1} = feval;
    if options.verbose
        fprintf('Before any iteration, function value: %f\n', feval);
    end
    
    for i = 1:options.MaxIterations
        % Update the guess range
        if sign(feval) == right_sign
            x_max = x;
        else
            x_min = x;
        end
        x = (x_max + x_min) / 2;
        feval = f(x);
        history.x{i + 1} = x;
        history.feval{i + 1} = feval;

        if abs(feval) < options.OptimalityTolerance
            stop_reason = 'The function value is smaller than the optimality tolerance.';
            break
        end
        
        if options.verbose
            fprintf('Iteration: %5d, Function value: %f, Solution: %f\n', i, feval, x);
        end
    end % end of iterations
    
    if options.verbose
        if i == options.MaxIterations
            stop_reason = 'The maximum number of iterations was reached.';
        end
        fprintf('Stopped after %d iterations because:\n%s\nFinal function value: %f, Solution: %f\n', i, stop_reason, feval, x);
    end
    stop_iter = i;
end % end of function