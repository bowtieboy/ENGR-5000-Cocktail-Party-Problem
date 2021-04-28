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

%% Add the speaker to the model

% Pass the audio, fs, and name to the model
speechSeparator.addNewSpeaker(audio.', fs, speaker_name);

% Save the model
save('speechSeparator.mat', 'speechSeparator');