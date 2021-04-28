clear;
clc;
%% Download files
url = 'http://ssd.mathworks.com/supportfiles/audio/CocktailPartySourceSeparation.zip';

downloadNetFolder = tempdir;
netFolder = fullfile(downloadNetFolder,'CocktailPartySourceSeparation');

if ~exist(netFolder,'dir')
    disp('Downloading pretrained network and audio files (5 files - 24.5 MB) ...');
    unzip(url,downloadNetFolder);
end

%% Visualize speech

% Load files, Fs is the same for both so only create on variable
[mSpeech,Fs] = audioread(fullfile(netFolder,"MaleSpeech-16-4-mono-20secs.wav"));
[fSpeech] = audioread(fullfile(netFolder,"FemaleSpeech-16-4-mono-20secs.wav"));

% Combine the sources
mSpeech = mSpeech/norm(mSpeech);
fSpeech = fSpeech/norm(fSpeech);
ampAdj  = max(abs([mSpeech;fSpeech]));
mSpeech = mSpeech/ampAdj;
fSpeech = fSpeech/ampAdj;
mix     = mSpeech + fSpeech;
mix     = mix ./ max(abs(mix));

% Visaulize the sources, as well as the mix
t = (0:numel(mix)-1)*(1/Fs);
figure(1)
subplot(3,1,1)
plot(t,mSpeech)
title("Male Speech")
grid on
subplot(3,1,2)
plot(t,fSpeech)
title("Female Speech")
grid on
subplot(3,1,3)
plot(t,mix)
title("Speech Mix")
xlabel("Time (s)")
grid on

% Visualize as a time-frequency graph
WindowLength  = 128;
FFTLength     = 128;
OverlapLength = 96;
win           = hann(WindowLength,"periodic");

figure(2)
subplot(3,1,1)
stft(mSpeech, Fs, 'Window', win,'OverlapLength',OverlapLength,...
     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
title("Male Speech");
subplot(3,1,2);
stft(fSpeech, Fs, 'Window', win, 'OverlapLength', OverlapLength,...
     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
title("Female Speech");
subplot(3,1,3);
stft(mix, Fs, 'Window', win, 'OverlapLength', OverlapLength,...
    'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
title("Mix Speech");

%% Source sepeartion using a Binary Time-Frequnecy Mask

% Ideal binary mask creation
P_M        = stft(mSpeech, 'Window', win, 'OverlapLength', OverlapLength,...
                 'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
P_F        = stft(fSpeech, 'Window', win, 'OverlapLength', OverlapLength,...
                 'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
[P_mix,F]  = stft(mix, 'Window', win, 'OverlapLength', OverlapLength,...
                  'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
binaryMask = abs(P_M) >= abs(P_F);

% Visualize the mask
figure(3)
plotMask(binaryMask, WindowLength - OverlapLength, F, Fs);

% Estimate source seperation speech using mask
P_M_Hard = P_mix .* binaryMask;
P_F_Hard = P_mix .* (1-binaryMask);

% Inverse the short-time fourier transform
mSpeech_Hard = istft(P_M_Hard , 'Window', win, 'OverlapLength', OverlapLength,...
                     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
fSpeech_Hard = istft(P_F_Hard , 'Window', win, 'OverlapLength', OverlapLength,...
                     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
                 
% Plot the results
figure(4)
subplot(2,2,1)
plot(t,mSpeech)
axis([t(1) t(end) -1 1])
title("Original Male Speech")
grid on

subplot(2,2,3)
plot(t,mSpeech_Hard)
axis([t(1) t(end) -1 1])
xlabel("Time (s)")
title("Estimated Male Speech")
grid on

subplot(2,2,2)
plot(t,fSpeech)
axis([t(1) t(end) -1 1])
title("Original Female Speech")
grid on

subplot(2,2,4)
plot(t,fSpeech_Hard)
axis([t(1) t(end) -1 1])
title("Estimated Female Speech")
xlabel("Time (s)")
grid on

%% Source sepeartion using a Soft Time-Frequnecy Mask

% Create the soft mask
softMask = abs(P_M) ./ (abs(P_F) + abs(P_M) + eps);

P_M_Soft = P_mix .* softMask;
P_F_Soft = P_mix .* (1-softMask);

mSpeech_Soft = istft(P_M_Soft, 'Window', win, 'OverlapLength', OverlapLength,...
                     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
fSpeech_Soft = istft(P_F_Soft, 'Window', win, 'OverlapLength', OverlapLength,...
                     'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
                 
% Plot the results
figure(5)
subplot(2,2,1)
plot(t,mSpeech)
axis([t(1) t(end) -1 1])
title("Original Male Speech")
grid on

subplot(2,2,3)
plot(t,mSpeech_Soft)
axis([t(1) t(end) -1 1])
title("Estimated Male Speech")
grid on

subplot(2,2,2)
plot(t,fSpeech)
axis([t(1) t(end) -1 1])
xlabel("Time (s)")
title("Original Female Speech")
grid on

subplot(2,2,4)
plot(t,fSpeech_Soft)
axis([t(1) t(end) -1 1])
xlabel("Time (s)")
title("Estimated Female Speech")
grid on

%% Prepare data for deep learning estimation

% Create training and validation datasets
maleTrainingAudioFile   = "MaleSpeech-16-4-mono-405secs.wav";
femaleTrainingAudioFile = "FemaleSpeech-16-4-mono-405secs.wav";
maleSpeechTrain   = audioread(fullfile(netFolder,maleTrainingAudioFile));
femaleSpeechTrain = audioread(fullfile(netFolder,femaleTrainingAudioFile));
L = min(length(maleSpeechTrain),length(femaleSpeechTrain));  
maleSpeechTrain   = maleSpeechTrain(1:L);
femaleSpeechTrain = femaleSpeechTrain(1:L);

maleValidationAudioFile   = "MaleSpeech-16-4-mono-20secs.wav";
femaleValidationAudioFile = "FemaleSpeech-16-4-mono-20secs.wav";
maleSpeechValidate   = audioread(fullfile(netFolder,maleValidationAudioFile));
femaleSpeechValidate = audioread(fullfile(netFolder,femaleValidationAudioFile));
L = min(length(maleSpeechValidate),length(femaleSpeechValidate));  
maleSpeechValidate   = maleSpeechValidate(1:L);
femaleSpeechValidate = femaleSpeechValidate(1:L);

% Scale the datasets to the same power
maleSpeechTrain   = maleSpeechTrain/norm(maleSpeechTrain);
femaleSpeechTrain = femaleSpeechTrain/norm(femaleSpeechTrain);
ampAdj            = max(abs([maleSpeechTrain;femaleSpeechTrain]));
maleSpeechTrain   = maleSpeechTrain/ampAdj;
femaleSpeechTrain = femaleSpeechTrain/ampAdj;

maleSpeechValidate   = maleSpeechValidate/norm(maleSpeechValidate);
femaleSpeechValidate = femaleSpeechValidate/norm(femaleSpeechValidate);
ampAdj               = max(abs([maleSpeechValidate;femaleSpeechValidate]));
maleSpeechValidate   = maleSpeechValidate/ampAdj;
femaleSpeechValidate = femaleSpeechValidate/ampAdj;

% Create mixed audio for training and valdiation
mixTrain = maleSpeechTrain + femaleSpeechTrain;
mixTrain = mixTrain / max(mixTrain);

mixValidate = maleSpeechValidate + femaleSpeechValidate;
mixValidate = mixValidate / max(mixValidate);

% Calculate training STFTs
WindowLength  = 128;
FFTLength     = 128;
OverlapLength = 128-1;
Fs            = 4000;
win           = hann(WindowLength,"periodic");

P_mix0 = stft(mixTrain, 'Window', win, 'OverlapLength', OverlapLength,...
              'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
P_M    = abs(stft(maleSpeechTrain, 'Window', win, 'OverlapLength', OverlapLength,...
              'FFTLength', FFTLength, 'FrequencyRange', 'onesided'));
P_F    = abs(stft(femaleSpeechTrain, 'Window', win, 'OverlapLength', OverlapLength,...
              'FFTLength', FFTLength, 'FrequencyRange', 'onesided'));

P_mix = log(abs(P_mix0) + eps);
MP    = mean(P_mix(:));
SP    = std(P_mix(:));
P_mix = (P_mix - MP) / SP;          

% Calculate validation STFTs
P_Val_mix0 = stft(mixValidate, 'Window', win, 'OverlapLength', OverlapLength,...
                 'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
P_Val_M    = abs(stft(maleSpeechValidate, 'Window', win, 'OverlapLength', OverlapLength,...
                 'FFTLength', FFTLength, 'FrequencyRange', 'onesided'));
P_Val_F    = abs(stft(femaleSpeechValidate, 'Window', win, 'OverlapLength', OverlapLength,...
                 'FFTLength', FFTLength, 'FrequencyRange', 'onesided'));

P_Val_mix = log(abs(P_Val_mix0) + eps);
MP        = mean(P_Val_mix(:));
SP        = std(P_Val_mix(:));
P_Val_mix = (P_Val_mix - MP) / SP;

% Check the distribution of the mixed dataset to ensure it is relatively
% smooth and normalized
figure(6)
histogram(P_mix,"EdgeColor","none","Normalization","pdf");
xlabel("Input Value");
ylabel("Probability Density");

% Compute the training and validation softmasks
maskTrain = P_M ./ (P_M + P_F + eps);
maskValidate = P_Val_M ./ (P_Val_M + P_Val_F + eps);

% Create windows from the training audio signals
seqLen        = 20;
seqOverlap    = 10;
mixSequences  = zeros(1 + FFTLength/2,seqLen,1,0);
maskSequences = zeros(1 + FFTLength/2,seqLen,1,0);

loc = 1;
while loc < size(P_mix,2) - seqLen
    mixSequences(:,:,:,end+1)  = P_mix(:,loc:loc+seqLen-1); %#ok
    maskSequences(:,:,:,end+1) = maskTrain(:,loc:loc+seqLen-1); %#ok
    loc                        = loc + seqOverlap;
end

%Create windows from the validation audio signals
mixValSequences  = zeros(1 + FFTLength/2,seqLen,1,0);
maskValSequences = zeros(1 + FFTLength/2,seqLen,1,0);
seqOverlap       = seqLen;

loc = 1;
while loc < size(P_Val_mix,2) - seqLen
    mixValSequences(:,:,:,end+1)  = P_Val_mix(:,loc:loc+seqLen-1); %#ok
    maskValSequences(:,:,:,end+1) = maskValidate(:,loc:loc+seqLen-1); %#ok
    loc                           = loc + seqOverlap;
end

% Reshape the training and validation signals
mixSequencesT  = reshape(mixSequences,    [1 1 (1 + FFTLength/2) * seqLen size(mixSequences,4)]);
mixSequencesV  = reshape(mixValSequences, [1 1 (1 + FFTLength/2) * seqLen size(mixValSequences,4)]);
maskSequencesT = reshape(maskSequences,   [1 1 (1 + FFTLength/2) * seqLen size(maskSequences,4)]);
maskSequencesV = reshape(maskValSequences,[1 1 (1 + FFTLength/2) * seqLen size(maskValSequences,4)]);

%% Create deep learning model to estimate the ideal soft mask

% Define the network
numNodes = (1 + FFTLength/2) * seqLen;

layers = [ ...
    
    imageInputLayer([1 1 (1 + FFTLength/2)*seqLen],"Normalization","None")
    
    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(6)
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(6)
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(0)

    regressionLayer
    
    ];

% Specify the training parameters
maxEpochs     = 3;
miniBatchSize = 64;

options = trainingOptions("adam", ...
    "MaxEpochs",maxEpochs, ...
    "MiniBatchSize",miniBatchSize, ...
    "SequenceLength","longest", ...
    "Shuffle","every-epoch",...
    "Verbose",0, ...
    "Plots","training-progress",...
    "ValidationFrequency",floor(size(mixSequencesT,4)/miniBatchSize),...
    "ValidationData",{mixSequencesV,maskSequencesV},...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.9, ...
    "LearnRateDropPeriod",1);

% Train the network
doTraining = false;
if doTraining
    CocktailPartyNet = trainNetwork(mixSequencesT,maskSequencesT,layers,options);
    save("CocktailPartyNet.mat", 'CocktailPartyNet');
else
    s = load("CocktailPartyNet.mat");
    CocktailPartyNet = s.CocktailPartyNet;
end

% Validate the network
estimatedMasks0 = predict(CocktailPartyNet,mixSequencesV);
estimatedMasks0 = estimatedMasks0.';
estimatedMasks0 = reshape(estimatedMasks0,1 + FFTLength/2,numel(estimatedMasks0)/(1 + FFTLength/2));

% Plot histogram of error between actual and predicted masks
figure(7);
histogram(maskValSequences(:) - estimatedMasks0(:),"EdgeColor","none","Normalization","pdf");
xlabel("Mask Error");
ylabel("Probability Density");

% Evaluate the mask
SoftMaleMask   = estimatedMasks0; 
SoftFemaleMask = 1 - SoftMaleMask;

P_Val_mix0 = P_Val_mix0(:,1:size(SoftMaleMask,2));
P_Male = P_Val_mix0 .* SoftMaleMask;

maleSpeech_est_soft = istft(P_Male, 'Window', win, 'OverlapLength', OverlapLength,...
                           'FFTLength', FFTLength, 'ConjugateSymmetric', true,...
                           'FrequencyRange', 'onesided');
maleSpeech_est_soft = maleSpeech_est_soft / max(abs(maleSpeech_est_soft));

% Visualize the output
range = (numel(win):numel(maleSpeech_est_soft)-numel(win));
t     = range * (1/Fs);

figure(8)
subplot(2,1,1)
plot(t,maleSpeechValidate(range))
title("Original Male Speech")
xlabel("Time (s)")
grid on

subplot(2,1,2)
plot(t,maleSpeech_est_soft(range))
xlabel("Time (s)")
title("Estimated Male Speech (Soft Mask)")
grid on