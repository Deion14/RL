
% this code will run the control with TD(0) with or without custom features


%choose name TD or MC

FEATURES=1
if FEATURES==0
    cw2_ex2
elseif FEATURES==1    
    cw2_ex2_bonus % Custom Features that are only the 3 features in front of the car
end