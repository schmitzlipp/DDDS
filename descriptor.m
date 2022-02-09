classdef descriptor < handle 
    properties
        E
        A
        B
        C
        D
        S
        P
        A1
        N
        B1
        B2
        C1
        C2
        q
        m
        r
        p
        n
        s
        ud
        Hud
        Hyd
        Hxd
        xd
        yd
        L
        contr
        obsv
    end
    
    methods
        function obj = descriptor(E,A,B,C,D,varargin)
            obj.E = E;
            obj.A = A;
            obj.B = B;
            obj.C = C;
            obj.D = D;
            obj.n = size(E,1);
            obj.m = size(B,2);
            obj.p =  size(C,1);
            if nargin > 5
                [obj.N,obj.A1,obj.q,obj.s,obj.S,obj.P] = qweierstrass(E,A,varargin{1});
            else
                [obj.N,obj.A1,obj.q,obj.s,obj.S,obj.P] = qweierstrass(E,A);
            end
            obj.r = obj.n - obj.q;
            B_ = obj.S*obj.B;
            C_ = obj.C*obj.P;
            obj.B1 = B_(1:obj.q,:);
            obj.B2 = B_(obj.q+1:end,:);
            obj.C1 = C_(:,1:obj.q);
            obj.C2 = C_(:,obj.q+1:end);
            
            %% Test R-controllability and R-observability
            if rank(ctrb(obj.A1,obj.B1)) == obj.q
                obj.contr = true;
            else
                obj.contr = false;
                disp('Warning: System not R-controllable!');
            end
            if rank(obsv(obj.A1,obj.C1)) == obj.q
                obj.obsv = true;
            else
                obj.obsv = false;
                disp('Warning: System not R-observable!');
            
            end  
        end
            
        
        function [x, u, y] = get_trajectory2(obj, u, x1_init)
            T = size(u,2)-obj.s;
            x1 = zeros(obj.q, T);
            x1(:,1) = x1_init;
            for j = 2:T
                x1(:,j) = obj.A1*x1(:,j-1) +  obj.B1*u(:,j-1);
            end
            
            x2 = zeros(obj.r, T);
            for j = 1:T
                for k = 1:obj.s
                    x2(:,j) = x2(:,j) - obj.N^k * obj.B2 * u(:,k+j);
                end
            end
            x = obj.P*[x1;x2];
            %y = obj.C1*x1 + obj.C2*x2 + obj.D*u(:,1:T);
            y = obj.C*x + obj.D*u(:,1:T);
            u = u(:,1:T);
        end
        
        function value = validate_trajectory(obj, x, u, y, varargin)
            value = true;
            if nargin >= 5
                eps = varargin{1};
            else
                eps = 1e-5;
            end
            
            err1 = vecnorm( obj.E*x(:,2:end) - (obj.A*x(:,1:end-1) + obj.B*u(:,1:end-1)) );
            err2 = vecnorm( y - (obj.C*x + obj.D*u) );
            
            if max(err1)+max(err2)>eps
                value = false;
                max(err1)+max(err2)
                disp('Warning: No trajectory!')
            end
        end
        
        function gen_model(obj, u, L)
            L_ = L + obj.s - 1 + obj.q;
            H = hankel(u, L_);
            if rank(H)<obj.m*L_
                disp('Warning: Order of persistent excitation insufficient!')
                disp(obj.m*L_-rank(H))
            end
            %[x, u, y] = obj.get_trajectory(u, x1_init);
            [x, u, y] = obj.get_trajectory(u);
            obj.Hud = hankel(u, L);
            obj.Hxd = hankel(x, L);
            obj.Hyd = hankel(y, L);
            obj.xd = x;
            obj.ud = u;
            obj.yd = y;
            obj.L = L;
        end  
        
        function [x, u, y] = predict_trajectory(obj, u_init, y_init, varargin)
        %% predict_trajectory(obj, u_init, y_init, u_stead, y_stead)
        if nargin > 4
            u_stead = varargin{1};
            y_stead = varargin{2};
        else
            u_stead = zeros(obj.m,1);
            y_stead = zeros(obj.p,1);
        end
        
        if nargin > 6
            u_shift = varargin{1};
            y_shift = varargin{2};
        else
            u_shift = zeros(obj.m,1);
            y_shift = zeros(obj.p,1);
        end
        %% check feasability
        L_ic = obj.s - 1 + obj.q;
        L_fs = 2 * obj.q + 3*obj.s - 2;
        L_end = obj.q + obj.s - 2;
        if obj.L < L_ic+L_fs
            error('Prediction horizon to short!')
        end
        
        %% truncate u_init, y_init
        u_init = u_init(:, end-L_end:end);
        y_init = y_init(:, end-L_end:end);
        
        Hud_1 = [ obj.Hud(1:L_ic*obj.m, :); obj.Hud(end-L_end*obj.m+1:end, :) ];
        Hyd_1 = [ obj.Hyd(1:L_ic*obj.p, :); obj.Hyd(end-L_end*obj.p+1:end, :) ];
        
        Hud_2 = obj.Hud(L_ic*obj.m+1:end, :);
        Hyd_2 = obj.Hyd(L_ic*obj.p+1:end, :);
        
        H_1 = [Hud_1; Hyd_1];
        H_2 = [Hud_2; Hyd_2];
        
        steady = [ repmat(u_stead+u_shift, obj.L-L_ic, 1); repmat(y_stead+y_shift, obj.L-L_ic, 1)];
        b = [reshape(u_init,[],1); repmat(u_stead, L_end, 1); reshape(y_init,[],1); repmat(y_stead, L_end, 1)];
        %% pre-optimization with soft-constraint
        w = 1e+3;
        [alpha_pre,fval,exitflag] = quadprog((H_2'*H_2)+ w*(H_1'*H_1), -steady'*H_2 -w*b'*H_1, [], [], [], [], [], [], []);
        %% optimization with initial value
        options = optimoptions('quadprog','Algorithm','active-set', 'MaxIterations', 200);
        [alpha,fval,exitflag] = quadprog((H_2'*H_2), -steady'*H_2, [], [], H_1, b, [], [], alpha_pre, options);
        %%alpha = alpha_pre;%(1:size(H_2,2));
        fprintf('fval %f; exitflag %d\n', fval, exitflag);
        disp(norm(H_1*alpha-b));
        x = reshape(obj.Hxd*alpha, obj.n,[]);
        y = reshape(obj.Hyd*alpha, obj.p, []);
        u = reshape(obj.Hud*alpha, obj.m, []);
        end
        
        function u = pe_input(obj, L)
            u = zeros(obj.m,(obj.m+1)*L-1);
            for j = 1:obj.m
                u(j,j*L) = 1;
            end    
        end
        
        function [] = auto_model(obj, L)
            L_ = L + obj.s - 1 + obj.q;
            u = obj.pe_input(L_);
            %[x, u, y] = obj.get_trajectory(u, zeros(obj.q,1));
            [x, u, y] = obj.get_trajectory(u);
            obj.Hud = hankel(u, L);
            obj.Hxd = hankel(x, L);
            obj.Hyd = hankel(y, L);
            obj.xd = x;
            obj.ud = u;
            obj.yd = y;
            obj.L = L;
        end
        
        function [x, u, y] = get_trajectory(obj, u, varargin)
            %% [x, u, y] = get_trajectory(obj, u, x_init)
            T = size(u,2);
            V = kron([eye(T),zeros(T,1)], -obj.A);
            V = V + kron([zeros(T,1),eye(T)], obj.E);
            U = reshape(obj.B*u,[],1);
            if nargin > 2
                V2 = [eye(obj.n), zeros(obj.n, obj.n*T)];
                V = [V;V2];
                rank(V)
                U = [U;varargin{1}];
            end
            X = linsolve(V,U);
            x = reshape(X, obj.n,[]);
            x =  x(:,1:end-1);
            y = obj.C*x + obj.D*u;
        end
        
        function [x, u, y] = mpc_state(obj,x, u, y, K, varargin)
            %% The state x is not used in the prediction step!
            if nargin > 5
                u_stead = varargin{1};
                y_stead = varargin{2};
            else
                u_stead = zeros(obj.m,1);
                y_stead = zeros(obj.p,1);
            end
            for k = 1:K
                [xp, up, yp] = obj.predict_trajectory(u, y, u_stead, y_stead);
                x = [x, xp(:,obj.q + obj.s)];
                y = [y, yp(:,obj.q + obj.s)];
                u = [u, up(:,obj.q + obj.s)];
            end
        end
            
        function [u, y] = mpc(obj, u, y, K, varargin)
            if nargin > 4
                u_stead = varargin{1};
                y_stead = varargin{2};
            else
                u_stead = zeros(obj.m,1);
                y_stead = zeros(obj.p,1);
            end
            for k = 1:K
                [~, up, yp] = obj.predict_trajectory(u, y, u_stead, y_stead);
                u = [u, up(:,obj.q + obj.s)];
                if nargin > 6
                    func = varargin{3};
                    y = [y, func(u,y)];
                else
                    y = [y, yp(:,obj.q + obj.s)];
                end
            end
        end
        
        function [X,U,Y] = steady_states(obj)
            W = null([obj.A-obj.E, obj.B]);
            X = W(1:obj.n,:);
            U = W(obj.n+1:end,:);
            Y = obj.C*X + obj.D*U;
        end
        
    end
end

function [H] = hankel(f, L)
    [m,n] = size(f);
    H = zeros(m*L, n-L+1);
    for j=1:n-L+1
        H(:,j) = reshape(f(:,j:j+L-1),[],1);
    end
end

function [N,A1,q,s,S,P] = qweierstrass(E,A, varargin)
    %% Calulation of the quasi-Weierstraß form via Wong sequences, see [1]
    %% [1] T. Berger, A. Ilchmannm, S. Trenn, The quasi-Weierstraß form for regular matrix pencils, Linear Algebra Appl 436 (2012), 4052–4069.

    n = size(A,1);

    if nargin >= 3
        lam = varargin{1};
    else
        lam = 0.0;
    end

    %% Test pencil for regularity
    kmax = 100;
    %condmax = 100;
    k = 1;
    while rank(A-lam*E)<n% || cond(A-lam*E)>condmax
        lam = rand(1)*100;
        k = k + 1;
        if k>kmax
            error('Error: Pencil possibly singular!')
        end
    end

    K = (A-lam*E)\E;

    %% calculate dimension q=dim(V^*)=n-dim(W^*)
    q = rank(K);
    q_old = n;
    i=1;
    while q_old>q
        i = i + 1;
        q_old = q;
        q = rank(K^i);
    end
    i = i-1;

    %% calculate matrices V,W such that im(V)=V^* and im(W)=W^*
    V = orth(K^i);
    W = null(K^i);


    %% calculate transfomration matrices S,P such that pencil S*(lam*E-A)*P is in quasi-Weierstraß from 
    S_inv = [E*V,A*W];
    S = inv(S_inv);
    P = [V,W];

    E_ = S_inv\E*P;
    A_ = S_inv\A*P;

    N = E_(q+1:end,q+1:end);
    A1 = A_(1:q,1:q);

    %% calculate nilpotency index s of N
    Npow = N;
    s = 1;
    while any(any(Npow)) && s<(n-q)
        Npow = Npow*N;
        s = s+1;
    end

end

