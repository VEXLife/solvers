classdef LinearEquationSolverTest < matlab.unittest.TestCase
    %% Generate a linear equation problem and test solvers

    properties
        A
        b
        x
    end
    
    properties (TestParameter)
        problemsettings = struct("m",128,"n",128,"tol",1e-1);
    end
    
    methods(TestClassSetup)
        % Shared setup for the entire test class

        function generateProblem(testCase)
            rng(1);

            % Generate a non-negative least squares problem
            L = randn(testCase.problemsettings.m, testCase.problemsettings.n);
            testCase.A = L' * L + 1e-5 * eye(testCase.problemsettings.n);
            testCase.x = randn(testCase.problemsettings.n, 1);
            testCase.b = testCase.A * testCase.x;
        end
    end

    methods
        function verifySolution(testCase, sol)
            testCase.verifyLessThan(norm(sol - testCase.x) / norm(testCase.x),...
                testCase.problemsettings.tol);
        end
    end
    
    methods(Test)
        % Test methods
        
        function jacobiTest(testCase)
            sol = jacobi_lsqr(testCase.A, testCase.b,...
                OptimalityTolerance=1e-4, MaxIterations=2000);
            verifySolution(testCase, sol);
        end

        function gaussSeidelTest(testCase)
            sol = gauss_seidel_lsqr(testCase.A, testCase.b,...
                OptimalityTolerance=1e-4, MaxIterations=2000);
            verifySolution(testCase, sol);
        end

        function sorTest(testCase)
            sol = sor_lsqr(testCase.A, testCase.b,...
                OptimalityTolerance=1e-4, MaxIterations=2000);
            verifySolution(testCase, sol);
        end
    end
    
end