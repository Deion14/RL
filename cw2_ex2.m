

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
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );


%% Initialising the state observation (state features) and setting up the
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

W= zeros(4, 5, 3);

%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.

alpha=0.001;
gamma=0.75;
decayrate = 0.999;
N=1000
mean_squared_error=zeros(N,1);

delta_episode=zeros(N,1);
delta=zeros(24,1);


for episode = 1:N
    
    
    %%
 %   rng(seed); % uncomment for same map
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
    %zeros(4,5)
    Prev_V_Hat=0;
    Prev_agentRewardSignal=0;
    
    squaredError=zeros(episodeLength,1);
    
    
    for i = 1:episodeLength
        alpha = alpha*decayrate;
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
        
        % "previous action:
        % Replace Q_test1 with W which are the weights of the problem
        for action = 1:3
            action_values(action) = ...
                sum ( sum( W(:,:,action) .* stateFeatures ) );
        end % for each possible action
        

        [~, agentCommandOld] = max(action_values);
          

        % E-Greedy 
        eGreedyPart=0.15;
        if rand(1)>1-eGreedyPart
          agentCommandOld = randi(3) ;
        end


         Q_hat_old=   action_values( agentCommandOld);
        % The $GridMap$ functions $getTransitions$ and $getReward$ act as the
        % problems transition and reward function respectively.
        %
        % Your agent might not know these functions, but your simulator
        % does! (How wlse would we get our data?)
        %
        % $actionMoveAgent$ can be used to simulate agent (the car) behaviour.
        
        %     [ possibleTransitions, probabilityForEachTransition ] = ...
        %         MDP.getTransitions( realAgentLocation, actionTaken );
        %     [ numberOfPossibleNextStates, ~ ] = size(possibleTransitions);
        %     previousAgentLocation = realAgentLocation;
        
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( agentCommandOld, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;
        
        %     MDP.getReward( ...
        %             previousAgentLocation, realAgentLocation, actionTaken )
        
        
        % "next action for t+1:
        % For improvement Replace Q_test1 with W which are the weights of the problem
        stateFeatures_new = MDP.getStateFeatures(realAgentLocation);
        for action = 1:3
            action_values(action) = ...
                sum ( sum( W(:,:,action) .* stateFeatures_new ) );
        end 
        
        [~, agentCommandNew] = max(action_values);
          

        % E-Greedy 
        eGreedyPart=0.15;
        if rand(1)>1-eGreedyPart
          agentCommandNew = randi(3) ;
        end


         Q_hat_new=   action_values( agentCommandNew);
        
        
        error=agentRewardSignal+gamma*Q_hat_new-Q_hat_old;
        squaredError(i)=error^2;
        Delta_W= alpha*(error)*stateFeatures;
        oldW=W;
        W(:,:,agentCommandOld)=W(:,:,agentCommandOld)+Delta_W;
        
          delta(i)= max(max(max(abs(W-oldW))));
        
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
        %  pause(0.15)
        
    end
    mean_squared_error(episode)= mean(squaredError);
     delta_episode(episode)= mean(delta);
    
    currentMap = MDP ;
    agentLocation = realAgentLocation ;
    
    Return;
    
    %  printAgentTrajectory
    % pause(1)
    
end % for each episode
W

plot(linspace(1,N,N),mean_squared_error)
legend('MSE')
figure 
plot(linspace(1,N,N),delta_episode)
legend('DeltaW sort of')