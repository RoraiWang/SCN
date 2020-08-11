%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stochastic Configuration Netsworks Class (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017
classdef SCN
    properties
        Name = 'Stochastic Configuration Networks';
        version = '1.0 beta';
        % Basic parameters (networks structure)
        L       % hidden node number / start with 1
        W       % input weight matrix
        b       % hidden layer bias vector
        Beta    % output weight vector
        % Configurational parameters
        r       % regularization parameter
        tol     % tolerance
        Lambdas % random weights range, linear grid search
        L_max   % maximum number of hidden neurons
        T_max   % Maximum times of random configurations
        % Else
        nB = 1 % how many node need to be added in the network in one loop
        verbose = 50 % display frequency
        COST = 0   % final error
    end
    %% Funcitons and algorithm
    methods
        %% Initialize one SCN model
        function obj = SCN(L_max, T_max, tol, Lambdas, r , nB)
            format long; 
            
            obj.L = 1;
  
            if ~exist('L_max', 'var') || isempty(L_max)
                obj.L_max = 100;
            else
                obj.L_max = L_max;
                if L_max > 5000
                    obj.verbose = 500; % does not need too many output
                end
            end
            if ~exist('T_max', 'var') || isempty(T_max)
                obj.T_max=  100;
            else
                obj.T_max = T_max;
            end
            if ~exist('tol', 'var') || isempty(tol)
                obj.tol=  1e-4;
            else
                obj.tol = tol;
            end
            if ~exist('Lambdas', 'var') || isempty(Lambdas)
                obj.Lambdas=  [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200];
            else
                obj.Lambdas = Lambdas;
            end
            if ~exist('r', 'var') || isempty(r)
                obj.r =  [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999];
            else
                obj.r = r;
            end
            if ~exist('nB', 'var') || isempty(nB)
                obj.nB = 1;
            else
                obj.nB = nB;
            end
        end
                
        %% inequality equation return the ksi
        function  [obj, ksi] = InequalityEq(obj, eq, gk, r_L)
            ksi = ((eq'*gk)^2)/(gk'*gk) - (1 - r_L)*(eq'*eq);
        end
        %% Search for {WB,bB} of nB nodes
        function [WB, bB, Flag] = SC_Search(obj, X, E0)
            Flag =  0;% 0: continue; 1: stop; return a good node /or stop training by set Flag = 1
            WB  = [];
            bB  = [];
            [~,d] = size(X); % Get Sample and feature number
            [~,m] = size(E0);
            % Linear search for better nodes
            C = []; % container of kesi
            for i_Lambdas = 1: length(obj.Lambdas)  % i index lambda
                Lambda = obj.Lambdas(i_Lambdas);    % Get the random weight and bias range
                % Generate candidates T_max vectors of w and b for selection
                WT = Lambda*( 2*rand(d, [obj.T_max])-1 ); % WW is d-by-T_max
                bT = Lambda*( 2*rand(1, [obj.T_max])-1 ); % bb is 1-by-T_max
                HT = logsig(bsxfun(@plus, X*WT, bT));
                for i_r = 1:length(obj.r)
                    r_L = obj.r(i_r); % get the regularization value
                    % calculate the Ksi value
                    for t = 1: obj.T_max % searching one by one
                        % Calculate H_t
                        H_t = HT(:,t);
                        % Calculate kesi_1 ... kesi_m
                        ksi_m = zeros(1, m);
                        for i_m = 1:m                            
                            eq = E0(:,i_m);
                            gk = H_t;
                            [obj, ksi_m(i_m)] = obj.InequalityEq(eq, gk, r_L);
                        end
                        Ksi_t = sum(ksi_m);                         
                        if min(ksi_m) > 0
                            C = cat(2,C, Ksi_t);
                            WB  = cat(2, WB, WT(:,t));
                            bB  = cat(2, bB, bT(:,t));
                        end
                    end
                    nC = length(C);
                    if nC >= obj.nB
                        break; % r loop
                    else
                        continue;
                    end
                end %(r)
                if nC >= obj.nB
                    break; % lambda loop
                else
                    continue;
                end
            end % (lambda)
            % Return the good node / or stop the training.
            if nC>= obj.nB
                [~, I] = sort(C, 'descend');
                I_nb = I(1:obj.nB);
                WB = WB(:, I_nb);
                bB = bB(:, I_nb);
                %HB = HB(:, I_nb);
            end
            if nC == 0 || nC<obj.nB % discard w b
                disp('End Searching ...');
                Flag = 1;
            end
        end
        
        %% Add nodes to the model
        function obj = AddNodes(obj, w_L, b_L)
            obj.W = cat(2,obj.W, w_L);
            obj.b = cat(2,obj.b, b_L);
            obj.L = length(obj.b);
        end
        
        %% Compute the Beta, Output, ErrorVector and Cost
        function [obj, O, E, Error] = UpgradeSCN(obj, X, T)
            H = obj.GetH(X);
            obj = obj.ComputeBeta(H,T);
            O = H*obj.Beta;
            E = T - O;
            Error =  Tools.RMSE(E);
            obj.COST = Error;
        end     
        
        %% ComputeBeta
        function [obj, Beta] = ComputeBeta(obj, H, T)
            Beta = pinv(H)*T;
            obj.Beta = Beta;
        end              
        %% Regression
        function [obj, per] = Regression(obj, X, T)             
            per.Error = [];
            E = T;
            Error =  Tools.RMSE(E);
            disp(obj.Name);
            while (obj.L < obj.L_max) && (Error > obj.tol)            
                if mod(obj.L, obj.verbose) == 0
                    fprintf('L:%d\t\tRMSE:%.6f \r', obj.L, Error );
                end
                [w_L, b_L, Flag] = SC_Search(obj, X, E);% Search for candidate node / Hidden Parameters
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                 
                [obj, ~ , E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all                
                %log
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
            end% while
            fprintf('#L:%d\t\tRMSE:%.6f \r', obj.L, Error );
            disp(repmat('*', 1,30));
        end
        
        %% Classification
        function [obj, per] = Classification(obj, X, T)            
            per.Error = []; % Cost function error
            per.Rate = [];  % Accuracy Rate
            E = T;
            Error =  Tools.RMSE(E);            
            Rate = 0;
            disp(obj.Name);
            while (obj.L < obj.L_max) && (Error > obj.tol)
                if mod(obj.L, obj.verbose) == 0
                    fprintf('L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
                end
                [w_L, b_L, Flag] = SC_Search(obj, X, E);
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                
                [obj, ~, E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all
                O = obj.GetLabel(X);
                Rate = 1- confusion(T',O');
                % Training LOG
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
                per.Rate = cat(2, per.Rate,  repmat(Rate, 1, obj.nB));
            end% while
            fprintf('#L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
            disp(repmat('*', 1,30));
        end

        %% Output Matrix of hidden layer
        function H = GetH(obj, X)
            H =  obj.ActivationFun(X);
        end
        % Sigmoid function
        function H = ActivationFun(obj,  X)
            H = logsig(bsxfun(@plus, X*[obj.W],[obj.b]));              
        end
        %% Get Output
        function O = GetOutput(obj, X)
            H = obj.GetH(X);
            O = H*[obj.Beta];
        end
        %% Get Label
        function O = GetLabel(obj, X)
            O = GetOutput(obj, X);
            O = Tools.OneHotMatrix(O);
        end
        %% Get Accuracy
        function [Rate, O] = GetAccuracy(obj, X, T)
            O = obj.GetLabel(X);
            Rate = 1- confusion(T',O');
        end
        %% Get Error, Output and Hidden Matrix
        function [Error, O, H, E] = GetResult(obj, X, T)
            % X, T are test data or validation data
            H = obj.GetH(X);
            O = H*(obj.Beta);
            E = T - O;
            Error =  Tools.RMSE(E);
        end
 
    end % methods
end % class
