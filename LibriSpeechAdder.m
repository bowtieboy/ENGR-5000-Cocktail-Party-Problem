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

% CD to director
cd('..\..\Diarization\LibriSpeech\train-clean-360');

% Number of libri-speakers to add
num_speakers = 10;
% How many of the files for each speaker should be added
desired_files = 1;

% Store data of speakers in struct
speakers = struct();
speaker_dirs = dir();
% Setup wait bar
w = waitbar(0, 'Speaker Number 1 Audio Being Stored...');
% Loop through each speaker
for s = 3 : num_speakers + 2
    wbstr = ['Speaker Number ', num2str(s - 2) ' Audio Being Stored...'];
    waitbar((s - 2) / num_speakers, w, wbstr);
    num_files = 1;
    num_data_points = 0;
    audio_data = struct();
    speakers(s - 2).name = ['libri', speaker_dirs(s).name];
    cd(speaker_dirs(s).name);
    chapters =  dir();
    % Loop through each chapter
    for c = 3 : length(chapters)
        cd(chapters(c).name);
        files = dir();
        % Loop through each file
        for f = 3 : desired_files + 2
            
            % If f is greater than the number of files in the directory,
            % exit the loop
            if (f > length(files))
                break
            end
            current_file = files(f).name;
            % Ensure the file is a flac
            if (contains(current_file, 'flac'))
                [audio, fs] = audioread(current_file);
                audio_data(num_files).audio = audio;
                audio_data(num_files).fs = fs;
                num_files = num_files + 1;
                num_data_points = num_data_points + length(audio);
            end
        end
        cd('..');
    end
    seconds = mod(num_data_points / fs, 60);
    minutes = floor((num_data_points / fs) / 60);
    speakers(s - 2).audio_data = audio_data;
    speakers(s - 2).num_files = num_files;
    disp(['Speaker ', speaker_dirs(s).name, ' had ', num2str(num_files),...
        ' files totaling ', num2str(minutes), ' minutes and ', num2str(seconds), ...
        ' seconds of data.']);
    cd('..');
end
close(w);
cd('..\..\..\Cocktail Party Problem\MATLAB');
%% Add the speakers to the model

% Preprocess the data and store it in the model
fbar = waitbar(0, 'Processing Audio Data For Speaker 1');
for s = 1 : length(speakers)
    wbstr = ['Processing Audio Data For Speaker ', num2str(s), '.'];
    waitbar(s / length(speakers), fbar, wbstr);
    disp(wbstr);
    speechSeparator.addNewSpeaker(speakers(s).audio_data, speakers(s).name);
end
close(fbar);
% Parition the data for the model
speechSeparator.partitionData();
% Save the model
save('speechSeparator.mat', 'speechSeparator');