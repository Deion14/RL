

%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;


%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance.
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly
% transitions into one of the above 3 states. If one of those states would
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid
% maps, complete with an initial location for your agent. (Also see the
% GridMap class).

noCarOnRowProbability = 0.8 ; % the probability that there is no car
% spawned for each row

seed = 1234;
% setting the seed for the random nunber generator

% Call this whenever starting a new episode:

%% Initialising the state observation (state features) and setting up the
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

W= zeros(4, 5, 3);;

%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.
alpha=1e-3;

decayrate = 0.99;
N=1000;
mean_squared_error=zeros(N,1);
rng(seed);
delta_episode=zeros(N,1);
delta=zeros(24,1)
for episode = 1:N
    
    %
    
    currentTimeStep = 0 ;
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
        blockSize, noCarOnRowProbability, ...
        probabilityOfUniformlyRandomDirectionTaken, rewards );
    
    
    currentMap = MDP ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
    
    % If you need to keep track of agent movement history:
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;
    
    realAgentLocation = agentLocation ; % The location on the full test map.
    Return = 0;
    
    HiststateFeatures=zeros(4,5,episodeLength);
    HistagentRewardSignal=zeros(24,1);
    Q_hat=zeros(24,1);
    actionHistory=zeros(24,1);
    alpha = max(1e-5,alpha*decayrate);
    for i = 1:episodeLength
        
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
        
        % save state features  by saving 1st value at bottom of vector
        HiststateFeatures(:,:,realAgentLocation(1,1)-1)=stateFeatures;
        
        % "previous action:
        for action = 1:3
            action_values(action) = ...
                sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
        end % for each possible action
        
        [~, action_old] = max(action_values);
        % save action  by saving 1st value at bottom of vector
        actionHistory(realAgentLocation(1,1)-1)=action_old;
        
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( action_old, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;
        % save reward from step  by saving 1st value at bottom of vector again
        HistagentRewardSignal(realAgentLocation(1,1))=agentRewardSignal;
        Return = Return + agentRewardSignal;
        
        % If you want to view the agents behaviour sequentially, and with a
        % moving view window, try using $pause(n)$ to pause the screen for $n$
        % seconds between each draw:
        
        [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
            MDP, realAgentLocation, blockSize );
        % $agentLocation$ is the location on the viewable grid map for the
        % simulation. It is used by $refreshScreen$.
        
        currentMap = viewableGridMap ; %#ok<NASGU>
        % $currentMap$ is keeping track of which part of the full test map
        % should be printed by $refreshScreen$ or $printAgentTrajectory$.
        
        %refreshScreen
        Prev_agentRewardSignal=agentRewardSignal;
        
        %  pause(0.15)
        
    end
    
    % create a G vector bu summing the future returns
    HistagentRewardsCum=cumsum(HistagentRewardSignal);
    currentMap = MDP ;
    agentLocation = realAgentLocation ;
    squaredError=zeros(episodeLength,1);
    
    %printAgentTrajectory
    
    for i =episodeLength:-1:1
        %Run MC algorithm dont have to take into account first visit or
        %every here since they will be the same
        
        % calculate q hat
        Q_hat=  sum(sum(W(:,:,actionHistory(i)) .* HiststateFeatures(:,:,i)));
        G=HistagentRewardsCum(i);
        % squaredError(i)=(G-Q_hat(i));
        squaredError(i)=(G-Q_hat)^2;
        
        
        Delta_W= alpha*(G-Q_hat)*HiststateFeatures(:,:,i);
        oldW =W;
        W(:,:,actionHistory(i))=W(:,:,actionHistory(i))+Delta_W;
        
        %find max differences between W
        delta(i)= max(max(max(abs(W-oldW))));
        
    end
    %take mean acros 24 steps
    mean_squared_error(episode)= mean(squaredError);
    delta_episode(episode)= mean(delta);
    Return;
    
    
    % pause(1)
    
end % for each episode

W(:,:,1:2)
plot(linspace(1,N,N),mean_squared_error)
plot(linspace(1,N,N),delta_episode)