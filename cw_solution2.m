
% this code will run the control with TD(0) with or without custom features
clc; clear all

%choose name control with or \ custom variables or MC
% runs 1000 episodes on the DIFFERENT maps
% Gives final weights
% plots two thigs an error measure and a convergence measure
% based on difference between W in updates


FEATURES=0
if FEATURES==0
    cw2_ex2
elseif FEATURES==1    
    cw2_ex2_bonus % Custom Features that are only the 3 features in front of the car
end