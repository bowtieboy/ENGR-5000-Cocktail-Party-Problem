clear;
clc;
%% Load the data

% Load the model if one exists. Otherwise create a new one
try
    load ('speechSeparator.mat');
catch
    disp('No SpeechSeparator model found in this directory, creating a new one.');
    speechSeparator = SpeechSeparator;
end

% Load the audio data
filename = 'C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Richard\Richard.m4a';
speaker_name = 'Richard Paquin';
[audio, fs] = audioread(filename);
audio_data = struct();
audio_data(1).audio = audio;
audio_data(1).fs = fs;
%% Add the speaker to the model

% Pass the audio, fs, and name to the model
speechSeparator.addNewSpeaker(audio_data, speaker_name);

% Parition the data
speechSeparator.partitionData();

% Save the model
save('speechSeparator.mat', 'speechSeparator');

%% Train the neural networks
speechSeparator.trainNetworks();

% Save the model
disp('Training completed, saving the models.');
save('speechSeparator.mat', 'speechSeparator');
disp('Saved filter masks!');
