% Author: Aman Sanghvi, z5116796
% Solution for MTRN4010, T1.2020, Project2.part1
% DemoEKF.m - version 2020.T1
% MRTN4010 - S1.2020
% Example using Extended Kalman Filter (EKF) for the localization problem, using LIDAR range observations.
% Read and analyze this progam; it wll be used on Project 2.


% Note#1:
% For each "LIDAR scan" we extract the detected landmarks and then we use their ranges to update the estimated 
% states (3DoF: 2D position and heading of the platform).  
% Even frequently using "just" one range (related to some landmark) the estimator is
% able to estimate the pose (position and heading, 3DoF)!!!
% Q: How is this possible?
% A: This is because in addition to the observation model we use the process model (which correlates the states in time) as well.
% As the robot moves then we obtain several observations from different poses (and at different times).
% This system is what is said to be "Observable" 
% (its internal states can be estimated by observing its "outputs", in this case we measure the robot's distance to certain landmark/landmarks).

% it seems that having just range (1D) and certain process model we would
% be able to estimate (x,y, Phi)  (3DoF)! 
% analogy/question: can we do stereo vision with a perfect process model and just 1 eye? 
% analogy/question: can we do stereo vision with a "good but not perfect" process model and just 1 eye? 
% answer: YES; this problem is called "mapping" (given, approximately, out position, we estimate the position of OOIs)


% Note#2: in this simulation we run the process model at certain frequency
% (dt=50ms), however we do not assume we get measurements (for the observations) at that sample
% rate (that situation would be great!, but it is not always possible). We assume observations have a slower sample rate. This
% is to show that even for a poor system (observing at low rate) the estimation process is still good and feasible. 


%Note #3:
% You can change many things in this simulation (that is the beauty of testing things under simulations):
% 1)  the geographical distribution and number of landmarks.
% 2)  the noise's standard deviations in the inputs of the kinematic model of the vehicle
% 3)  the noise's standard deviations of range measurements
% 4)  the sampling rate of the scanner
% 5)  the inputs to the process (speed and angular rate)
% and a lot of things for experimenting!


% You will be requested, in a subsequent project, to 
%  1) Modify the update stage of the EKF for accepting bearing observations.
%  2) Adapt the Q matrix used in the prediction step, for properly
%  considering the noise that pollutes the model's inputs.



% Note4:  IMPORTANT
% a) You will need to modify the EKF update stage in order to deal with the
% cases:   Bearing Only and Range+Bearing (as required in a subsequent project).

% b) you need to adapt the Q matrix in the prediction stage, in order to
% consider the noise that pollutes the inputs of the process model (speed
% and gyroZ). Follow the approach presented in the lecture notes, in file
% "MTRN4010_L06_Noise_in_the_inputs_of_ProcessModel.pdf"

%---------------------------------

% In this problem we have a kinematic model of the platform ( our "process
% model"). The inputs, u(t), of the model are 2 :
%    a)  platform's speed (measured at the middle point between the back wheels) 
%    b)  Wz (yaw/heading angular rate), measured through  IMU's gyros.


% by J. Guivant for MRTN4010's students, Session T1.2020



function main()


% .....................................................
% Here I define the magnitude of the "noises", which I assume are present, 
% polluting mesurements, models, etc.  
% (we are simulating those noises, so we can try whatever we want.) 

% Standard deviation of the error in the angular rate sensor. 
stdDevGyro = 2*pi/180;        
% 2 degrees/second , standard deviation of the gyros' noise
% you have seen, when playing with our real IMU's data that, after removing the
% bias, the random error present in the gyros' measurements should be lower than 2 deg/sec.
% so, this one seems realistic.

% Standard deviation of the error in the speed's measurements
stdDevSpeed = 0.15;   % We simulate a lot of error!  (very difficult case). 
% Speed-meter's error = 0.15m/sec; actually, a low quality speed sensor! (it
% may be the case, in cartain terrains in which platform's traction is not good.)

% ... errors in the range measurements (25cm, standard dev.)
sdev_rangeMeasurement = 0.25 ;          % std. of noise in range measurements. 0.25m
% this is more than the error you would have with our laser scanner.
sdev_angleMeasurement = pi/360; % half a degree.

% .....................................................

% some parameters, for the simulation context.
Dt=0.05 ;                       % "sample time", 50ms
Li = 5000 ;                     % "experiment" duration, in iterations 
DtObservations=0.250 ;          % laser "sample time" (for observations), 4Hz, approximately


% .....................................................
% Here, I create a virtual map, because this program works on a simulated process.
% (in a real case you would have a real map)

% How many landmarks do you want to use?  
n_usedLanmarks = 4 ;    %it must be : 0 < n_usedLanmarks <5, but you can modify the function "CreateSomeMap()" to extend it.
% just to create some landmarks
% if you specify "n_usedLanmarks = 0", you would not have observations
% (i.e. no updates, just predictions )

global NavigationMap;
NavigationMap = CreateSomeMap(n_usedLanmarks) ;  %creates a artificial map!
% ................................................


% In variables "Xe" and "P" :These are the EKF ESTIMATES (Expected value and covariance matrix)
% Initial conditions of the estimates (identical to the real ones, as in
% the lab)( I Assume we know the initial condition of the system)
estimatesVector = [ 0; 0; pi/2 ] ; 
P = zeros(3,3) ;            % initial quality --> perfect (covariance =zero )
% Why perfect? (BECAUSE in this case we DO ASSUME we know perfectly the initial condition)

% These are the "open-loop" dead reckoning ESTIMATES
deadReckoningVector = [ 0; 0;pi/2 ] ;

% Some buffers to store the intermediate values during the experiment (so we can plot them, later)
Xreal_History= zeros(3,Li) ;
Xe_History= zeros(3,Li) ;
XeDR_History= zeros(3,Li) ;

% .....................................................
% I assume that every time we apply the process model to predict the evolution of the system for a 
% perdiod of time of Dt (50ms) we introduce uncertainty of 0.01m standard deviation on X, 
% similar uncertainty on Y and 1 degree (a lot!) on the heading's estimation
% Q  =zeros(3,3) ;  Q(1,1) = (0.01)^2 ; Q(2,2) =Q(1,1)  ; Q(3,3) = (1*pi/180)^2 ;
% Although you can use this proposed Q, it can be improved. Read
% "MTRN4010_L06_Noise_in_the_inputs_of_ProcessModel.pdf" in order to implement a good refinement. 

Q = diag( [(0.02)^2 ,(0.02)^2 , (1*pi/180)^2]) ;
% Q matrix. Represent the covariance of the uncertainty about the process model.
Pu = diag([stdDevSpeed^2, stdDevGyro^2]);
% .....................................................


time=0 ;
% initialize the simulator of process (the "real system").
InitSimulation(stdDevSpeed,stdDevGyro,sdev_rangeMeasurement,DtObservations);
% (BTW: this is just a simulation to replace the real system, because we,
% for the moment, do not have the real system. )

clc();
disp('Running full simulation, OFF line. Please wait...');

% .....................................................    
for i=1:Li,     % loop
    
    %pause(dt) ;   % NO delay, I want to run this in a fast off-line SIMULATION, so ASAP.
    
    time = time+Dt ;    
    SimuPlatform(time,Dt);      % because NOW I do not have a real system, I simulate it.


    % I ask about the inputs to the Process Model.
    % The inputs I will be told are polluted versions of the real inputs.
    [Noisy_speed,Noisy_GyroZ]=GetProcessModelInputs();
    % in the real case we should measure the real inputs (gyros and speedmeter)

    % run our "open-loop" estimation (just dead-reckoning)
    deadReckoningVector = RunProcessModel(deadReckoningVector,Noisy_speed,Noisy_GyroZ,Dt);
    
    
    % -------- THIS IS THE ACTUAL EKF! ------------------------

    % ------ EKF's prediction: applies the prediction step (i.e. Porcess model). 
    % Estimate new covariance, associated to the state after prediction
    % First , I evaluate the Jacobian matrix of the process model (see lecture notes), at X=X(k|k).
    % You should write the analytical expression on paper to understand the following line.
    J = [[1, 0, -Dt*Noisy_speed*sin(estimatesVector(3))]; [0, 1, Dt*Noisy_speed*cos(estimatesVector(3))]; [ 0, 0, 1 ]] ; 
     
    % ATTENTION: we need, in our case, to propose a consistent Q matrix (this is part of your assignment!)
        
    % And, here, we calculate the predicted expected value. 
    estimatesVector = RunProcessModel(estimatesVector,Noisy_speed,Noisy_GyroZ,Dt);
    %  Xe(k+1|k) = f( Xe(k|k), u(k) )    % "Xe" means expected value.
    % in our case, we do not perfectly know the inputss u(k), but me measure them (which are noisy)
    % note: we reuse the same variable Xe (we do not need to remember Xe(k|k), so we overwrite it.)
    
    % then I calculate the new covarience, after the prediction P(K+1|K) = J*P(K|K)*J'+F*Pu*F'+Q;
    F = [Dt*cos(estimatesVector(3)) 0; Dt*sin(estimatesVector(3)) 0; 0 Dt];
    P = J*P*J' + F*d*F'+ Q;
    
    % so, here/now, the variable "Xe" contains "X^(k+1|k)"   ("X^" means "X hat", i.e. predicted expected value of X(k+1) )
    % and the variable "P" is "P(k+1|k)".
        
   % The predition step, for this iteration, is done.

    % .. Get range measuremens, if those are available.
    [nDetectedLandmarks, MeasuredRanges, MeasuredAngles, IDs]=GetObservationMeasurements();
    
    % if measurements are avaiable ==> we perform the related updates.
    if nDetectedLandmarks>0,     % any laser data and detected landmarks?
     
        % Because there are available obsevations ==> I perform EKF update\updates.
   
        % --------------- EKF update (observations)
        % Because this observation function is non-linear--> we need to get the Jacobians of h(X).
        % Jacobian for range only measurements (evaluated at the current expected value -->Xe)    
        
        % sqrt( (xi-x)^2+(yi-y)^2 ) for all the seen landmarks. I will need
        % this term for the next calculations.
        
        % for the sake of simplicity in this example, we perform an EKF update for each of the observations.
        for u=1:nDetectedLandmarks,
         
            ID = IDs(u);            % landmark ID?    (in "real life", this is provided by the "data association")
            
            % some auxiliary variables.
            d  = 0.46;
            x = estimatesVector(1);
            y = estimatesVector(2);
            phi = estimatesVector(3);
            eX = (NavigationMap.landmarks(ID,1) - x - d*cos(phi));      % (xu-x)
            eY = (NavigationMap.landmarks(ID,2) - y - d*sin(phi));      % (yu-y)
            eDist = sqrt(eX*eX + eY*eY); %   so : sqrt( (xu-x)^2+(yu-y)^2 )
            ePhi = atan2(eY,eX) - phi;
        
            % here is it. "H". I reuse some previous calculations.
            H = [-eX/eDist, -eY/eDist, (sin(ePhi)*d*eX - cos(ePhi)*d*eY)/eDist; ...
                 eY/(eDist^2), -eX/(eDist^2), (-d*(sin(ePhi)*eY + cos(ePhi)*eX)/(eDist^2)-1)]; % -d*(sin(ePhi)*eY/eDist^2 + cos(ePhi)*eX/(eDist^2))
             % Jacobian of h(X); size 2x3
             % see/read the lecture notes about EKF applied for implementing the localizer.   
            
            % the expected distances to this landmark ( "h(Xe)" )
            % the "Expected output".
            ExpectedRange = eDist;   % just a coincidence: we already calculated them for the Jacobian, so I reuse it. 
            ExpectedAngle = wrapToPi(ePhi);
        
            % Evaluate residual (innovation)  "Y-h(Xe)" 
            %(measured output value - expected output value)
            z  = [MeasuredRanges(u) - ExpectedRange; wrapToPi(MeasuredAngles(u) - ExpectedAngle)] ;      

            % ------ covariance of the noise/uncertainty in the measurements
            R = diag([sdev_rangeMeasurement*sdev_rangeMeasurement*4, sdev_angleMeasurement*sdev_angleMeasurement*4]) ;
            % I multiply by 4 because I want to be conservative and assume
            % twice the standard deviation that I believe does happen.
        
            % Some intermediate steps for the EKF (as presented in the lecture notes)
            S = R + H*P*H';
            iS=inv(S);                 % iS = inv(S) ;   % in this case S is 1x1 so inv(S) is just 1/S
            K = P*H'*iS ;           % Kalman gain
            % ----- finally, we do it...We obtain  X(k+1|k+1) and P(k+1|k+1)
            
            estimatesVector = estimatesVector+K*z;       % update the  expected value
            P = P-K*H*P;       % update the Covariance % i.e. "P = P-P*H'*iS*H*P"  )
            % -----  individual EKF update done ...
        
            % Loop to the next observation based on available measurements..
        end;  
       
       % Here, all available EKF updates have been done.
       % which means that the variable Xe contains X^(k+1|k+1); and P is P(k+1|k+1); the
       % expected value and covariance matrix, respectively, of the POSTERIOR PDF about X(k+1)
       
       
    end;  
     
     % -------- store some current variables, to plot later, at the end. 
     Xreal_History(:,i) = GetCurrentSimulatedState() ;
     Xe_History(:,i)    = estimatesVector ;
     XeDR_History(:,i)  = deadReckoningVector ;

end ;                           % end of while loop



fprintf('Done. Showing results, now..\n');
% now, we can see the resutls.
% PLOT some results. 
SomePlots(Xreal_History,Xe_History,XeDR_History,NavigationMap) ;


return ;        



% =========================================================================
% --- THIS IS THE PROCESS MODEL of MY SYSTEM. (it is a Kinemetic model)
    
function Xnext=RunProcessModel(X,speed,GyroZ,dt) 
    Xnext = X + dt*[ speed*cos(X(3)) ;  speed*sin(X(3)) ; GyroZ ] ;
return ;


% =========================================================================
% ========== Simulation functions - (used for simulation of "real" platform and
% for the process model of the EKF.
% When we apply the EKF on a real case, we do NOT need this part.


function [ranges, angles, IDs] = GetMeasurementsFomNearbyLandmarks(X,map)
    
    if map.nLandmarks>0,
        dx= map.landmarks(:,1) - X(1) ;
        dy= map.landmarks(:,2) - X(2) ;
        ranges = sqrt((dx.*dx + dy.*dy)) ;
        angles = atan2(dy, dx) - X(3);
        IDs = [1:map.nLandmarks];
    else,
        IDs=[];ranges=[];angles=[];
    end;
    % I simulate I measure/detect all the landmarks, however there can be
    % cases where I see just the nearby ones.
    
return ;


% here I propose some speed and angular rate inputs. 
% in real cases, they do happen, we do not propose them.
function [speed,GyroZ] = SimuControl(X,t)
    speed = 2 ;                                         % cruise speed, 2m/s  ( v ~ 7km/h)
    GyroZ = 3*pi/180 + sin(0.1*2*pi*t/50)*.02 ;         % some crazy driver moving the steering wheel...
return ;


% here I propose some map of landmarks. 
% in real cases, we do not create it, synthetically, like here.
function map = CreateSomeMap(n_used)
    n_used = max(0,min(4,n_used));      % accepts no less than 1, no more than 4. 
    
    landmarks = [  [ -40,0 ];[ -0,-20 ];[ 10,10 ] ;[ 30,10 ]  ] ;   
    % you may modify this list, in case you want to add more landmarks to the navigation map.
    
    map.landmarks = landmarks(1:n_used,:) ;
    map.nLandmarks = n_used ;
return ;





function InitSimulation(stdDevSpeed,stdDevGyro,sdev_rangeMeasurement,DtObservations)
    global ContextSimulation;
    ContextSimulation.Xreal = [ 0; 0;pi/2 ] ;     % [x;y;phi]
    ContextSimulation.stdDevSpeed = stdDevSpeed;
    ContextSimulation.stdDevGyro = stdDevGyro;
    ContextSimulation.Xreal = [0;0;pi/2];
    ContextSimulation.speed=0;
    ContextSimulation.GyroZ=0;
    ContextSimulation.sdev_rangeMeasurement=sdev_rangeMeasurement;
    ContextSimulation.DtObservations=DtObservations;
    ContextSimulation.timeForNextObservation= 0;
    ContextSimulation.CurrSimulatedTime=0;
return;


function [Noisy_speed,Noisy_GyroZ]=GetProcessModelInputs()
    % .....................................................
    % add noise to simulate real conditions
    % WHY? to be realistic. When we measure things the measurements are polluted with noise, So I simulated that situation by adding random
    % noise to the perfect measurements (the ones I get from the simulated "real" platform.
    global ContextSimulation;
    Noisy_speed =ContextSimulation.speed+ContextSimulation.stdDevSpeed*randn(1) ;
    Noisy_GyroZ =ContextSimulation.GyroZ+ContextSimulation.stdDevGyro*randn(1);
return;


% NOTE: in a subsequent project, you will apply the EKF for estimating the states of the real
% system. In such cases You DO NOT ADD ANY NOISE to the real
% measurements!!!!. We add noise, here, because we want to simulate a real
% case, always having noise polluting our measurements.


function  SimuPlatform(time,Dt)
    global ContextSimulation;    
    
    % simulate some crazy driver for the car..
    [ContextSimulation.speed,ContextSimulation.GyroZ] = SimuControl(ContextSimulation.Xreal,time) ;      % read kinematic model inputs, ideal ones
    % .........................................
    % simulate one step of the "real system":  Xreal(time)
    ContextSimulation.Xreal = RunProcessModel(ContextSimulation.Xreal,ContextSimulation.speed,ContextSimulation.GyroZ,Dt) ;
    ContextSimulation.CurrSimulatedTime = ContextSimulation.CurrSimulatedTime+Dt;
return;


function [nDetectedLandmarks,MeasuredRanges, MeasuredAngles,IDs]=GetObservationMeasurements(map)
    global ContextSimulation NavigationMap;       
   
    if ContextSimulation.CurrSimulatedTime<ContextSimulation.timeForNextObservation,
        % no measurements of outputs at this time.
        nDetectedLandmarks=0;
        MeasuredRanges=[];
        MeasuredAngles=[];
        IDs=[];
        return;
    end;
        
    ContextSimulation.timeForNextObservation = ContextSimulation.timeForNextObservation+ContextSimulation.DtObservations;
    
    % get simulated range measurements (actual system), in this case the
    % simulated platform.
        [RealRanges, RealAngles, IDs] = GetMeasurementsFomNearbyLandmarks(ContextSimulation.Xreal,NavigationMap) ;
                % ...................................................
        nDetectedLandmarks = length(RealRanges) ;
      
        
        if (nDetectedLandmarks<1)       % no detected landmarks...
            MeasuredRanges=[]; MeasuredAngles=[]; IDs=[];    return ; 
        end;
        
        
        % Here I corrupt the simulated measurements by adding some random noise (to simulate the noise that would be present in the reality)
        % this is the noise:
        noiseInMeasurements= ContextSimulation.sdev_rangeMeasurement*randn(size(RealRanges));
        % here I add it to the perfect ranges' measurements
        MeasuredRanges = RealRanges +  noiseInMeasurements;
        % so MeasuredRanges are the measurements polluted with
        % noise. I get the "perfect measurements" from the simulated
        % platform.
        % in real life they arrive already polluted!
        MeasuredAngles = RealAngles; % Unpolluted.
        
        
 return;

 
    function X=GetCurrentSimulatedState()
        global ContextSimulation;
        X=ContextSimulation.Xreal;
        
        
     return;   

% -------------end simulation functions -----------------------------------------------


% ====================================================
% --- This is JUST for ploting the results
function SomePlots(Xreal_History,Xe_History,Xdr_History,map) ;



figure(2) ; clf ; hold on ;
plot(Xreal_History(1,:),Xreal_History(2,:),'k') ;
plot(Xe_History(1,:),Xe_History(2,:),'r') ;
plot(Xdr_History(1,:),Xdr_History(2,:),'g') ;
if (map.nLandmarks>0),   
    plot(map.landmarks(:,1),map.landmarks(:,2),'*r') ;
    legend({'Real path','EKF Estimated path','DR Estimated path','Landmarks'});
else
    legend({'Real path','EKF Estimated path','DR Estimated path'});
end;    





% show vectors, for visualizing heading, at a small number of points.
ii = [1:225:length(Xe_History(1,:))] ;
m=10;
quiver(Xreal_History(1,ii),Xreal_History(2,ii),m*cos(Xreal_History(3,ii)),m*sin(Xreal_History(3,ii)),'b','AutoScale','off','Marker','o' ) ;
quiver(Xe_History(1,ii),Xe_History(2,ii),m*cos(Xe_History(3,ii)),m*sin(Xe_History(3,ii)),'r','AutoScale','off','Marker','+') ;
quiver(Xdr_History(1,ii),Xdr_History(2,ii),m*cos(Xdr_History(3,ii)),m*sin(Xdr_History(3,ii)),'m','AutoScale','off','Marker','o' ) ;




axis equal ;

title('Path') ;
zoom on ; grid on; box on;

% --------- plot errors between EKF estimates and the real values
figure(3) ; clf ; 
Xe=Xe_History;
subplot(311) ; plot(Xreal_History(1,:)-Xe(1,:)) ;ylabel('x-xe (m)') ;
title('Performance EKF') ;
subplot(312) ; plot(Xreal_History(2,:)-Xe(2,:)) ;ylabel('y-ye (m)') ;
subplot(313) ; plot(180/pi*(Xreal_History(3,:)-Xe(3,:))) ;ylabel('heading error (deg)') ;

figure(4) ; clf ; 
Xe=Xdr_History;
subplot(311) ; plot(Xreal_History(1,:)-Xe(1,:)) ;ylabel('x-xe (m)') ;
title('Performance Dead Reckoning (usually, not good)') ;
subplot(312) ; plot(Xreal_History(2,:)-Xe(2,:)) ;ylabel('y-ye (m)') ;
subplot(313) ; plot(180/pi*(Xreal_History(3,:)-Xe(3,:))) ;ylabel('heading error (deg)') ;
Xe=[];

return ;
% -------------------------------------------------------------------------


%  Jose Guivant -  AAS -Session 1.2020
%  Questions:     email to : j.guivant@unsw.edu.au


% -------------------------------------------------------------------------


