classdef LinearEquationSolverTest < matlab.unittest.TestCase
    %% Generate a non-negative linear equation problem and test solvers

    properties
        A
        b
        x
    end
    
    properties (TestParameter)
        problemsettings = struct("m",4096,"n",4096,"tol",1e-1);
    end
    
    methods(TestClassSetup)
        % Shared setup for the entire test class

        function generateProblem(testCase)
            rng(1);

            % Generate a non-negative least squares problem
            no_rx_antennas = 8;
            no_tx_horizontal_antennas = 16;
            no_tx_vertical_antennas = 8;
            no_tx_horizontal_beams = 32;
            no_tx_vertical_beams = 16;
            U = dftmtx(no_rx_antennas) / sqrt(no_rx_antennas);
            v_h = dftmtx(no_tx_horizontal_beams) / sqrt(no_tx_horizontal_antennas);
            v_h = v_h(1:no_tx_horizontal_antennas, :);
            v_v = dftmtx(no_tx_vertical_beams) / sqrt(no_tx_vertical_antennas);
            v_v = v_v(1:no_tx_vertical_antennas, :);
            V = kron(v_h, v_v);

            t_a = abs(U' * U) .^2;
            t_f = abs(V' * V) .^2;
            testCase.A=kron(t_f',t_a);
            testCase.x=rand(testCase.problemsettings.n, 1) * 200;
            testCase.x(testCase.x>1)=0; % Make the solution sparse
            testCase.b=testCase.A*testCase.x;
        end
    end

    methods
        function verifySolution(testCase, sol)
            testCase.verifyGreaterThanOrEqual(sol, 0);
            testCase.verifyLessThan(norm(sol - testCase.x) / norm(testCase.x),...
                testCase.problemsettings.tol);
        end
    end
    
    methods(Test)
        % Test methods
        
        function projectedGradientDescentTest(testCase)
            sol = pgd_lsqnonneg(testCase.A, testCase.b);
            verifySolution(testCase, sol);
        end

        function quadprogToLeastSquareWrapperTest(testCase)
            sol = quadprog_to_lsq_wrapper(testCase.A, testCase.b, @pgd_quadprognonneg);
            verifySolution(testCase, sol);
        end

        function multiplicativeUpdateTest(testCase)
            sol = quadprog_to_lsq_wrapper(testCase.A, testCase.b, @multipupd_quadprognonneg);
            verifySolution(testCase, sol);
        end

        function fixedPointIterationKLDivergenceTest(testCase)
            sol = fpi_kldivergence(testCase.A, testCase.b);
            verifySolution(testCase, sol);
        end

        function fixedPointIterationLeastSquareTest(testCase)
            sol = fpi_lsqnonneg(testCase.A, testCase.b);
            verifySolution(testCase, sol);
        end

        function gradientDescentKLDivergenceTest(testCase)
            sol = gd_kldivergence(testCase.A, testCase.b);
            verifySolution(testCase, sol);
        end
    end
    
end