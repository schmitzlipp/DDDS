clear all
clc

E = [0,0,1,0;1,2,0,2;2,3,1,3;1,2,0,2];
A = [1,1,0,2;0,2,1,1;1,4,2,3;-1,1,1,0];
B = [-1;2;2;3];
C = [1,2,1,2;0,1,0,1;1,2,1,1;2,2,1,2];
D = zeros(4,1);

sys = descriptor(E,A,B,C,D,0.0);

L = 11;
%T = 30;

%ud = (rand(sys.m,T)-0.5)*2;
%sys.gen_model(ud, L);
sys.auto_model(L)
[X,U,Y] = sys.steady_states();


ug = (rand(sys.m,11)-0.5)*2;
[xg,ug,yg] = sys.get_trajectory(ug);

[xp,up,yp] = sys.mpc_state(xg,ug,yg,20,0.0,[-10;0;0;-10]);
sys.validate_trajectory(xp,up,yp,1e-5)
%pause
[xp,up,yp] = sys.mpc_state(xp,up,yp,20,0.0,[15;0;0;15]);
sys.validate_trajectory(xp,up,yp,1e-5)
%pause
[xp,up,yp] = sys.mpc_state(xp,up,yp,20,0.0,[-5;0;0;-5]);
sys.validate_trajectory(xp,up,yp,1e-5)

figure();
plot(yp');
figure();
plot(up')
figure();
plot(xp')
