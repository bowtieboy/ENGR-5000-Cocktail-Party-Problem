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
filename = 'C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Matt\bible.flac';
speaker_name = 'Matthew Lima';
[audio, fs] = audioread(filename);

%% Get the output of the network

speechSeparator.separateSpeech(audio, fs, speaker_name);