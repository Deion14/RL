
% this code will run the TD(0) script or the MC script

clc; clear all

%choose name TD or MC
% runs 1000 episodes on the DIFFERENT maps
% Gives final weights
% plots two thigs an error measure and a convergence measure
% based on difference between W in updates
ALGORITHM=0
if ALGORITHM==0
    cw2_TD % TD script
else    
    cw2_MC % MC script
end