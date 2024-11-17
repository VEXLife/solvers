classdef NonnegativeLinearEquationSolverTest < matlab.unittest.TestCase
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
            A_row = [1 rand(1,testCase.problemsettings.n - 1) * 200];
            A_row(abs(A_row)>1)=0;
            A_col = [1 rand(1,testCase.problemsettings.m - 1) * 200];
            A_col(abs(A_col)>1)=0;
            testCase.A=toeplitz(A_col, A_row);
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
            sol = pgd_lsqnonneg(testCase.A, testCase.b,...
                OptimalityTolerance=1e-4);
            verifySolution(testCase, sol);
        end

        function quadprogToLeastSquareWrapperTest(testCase)
            sol = quadprog_to_lsqr_wrapper(testCase.A, testCase.b,...
                @pgd_quadprognonneg, 0, OptimalityTolerance=1e-4);
            verifySolution(testCase, sol);
        end

        function multiplicativeUpdateTest(testCase)
            sol = quadprog_to_lsqr_wrapper(testCase.A, testCase.b,...
                @multipupd_quadprognonneg, 0, OptimalityTolerance=1e-4);
            verifySolution(testCase, sol);
        end

        function fixedPointIterationKLDivergenceTest(testCase)
            sol = fpi_kldivergence(testCase.A, testCase.b,...
                OptimalityTolerance=1e-3);
            verifySolution(testCase, sol);
        end

        function fixedPointIterationLeastSquareTest(testCase)
            sol = fpi_lsqnonneg(testCase.A, testCase.b,...
                OptimalityTolerance=1e-3);
            verifySolution(testCase, sol);
        end

        function gradientDescentKLDivergenceTest(testCase)
            sol = gd_kldivergence(testCase.A, testCase.b,...
                OptimalityTolerance=1e-3);
            verifySolution(testCase, sol);
        end
    end
    
end