classdef (ConstructOnLoad) SpeechSeparator < handle
    
    % Public properties
    properties (SetAccess = public)
        
        % All speech data that will be used by the model
        speech_data = struct();
        speech_data_size = 0;
        % These two values MUST match those from SpeechProcessing
        window_size = 1;
        window_overlap = 0;
        % Used for filtering the audio
        bandpass_filter;
        % Training and test data sets for machine learning
        training_data = struct();
        testing_data = struct();
        dataset_size = 0;
        training_percentage = 0.75;
        % Struct of soft masks for each speaker
        filter_masks = struct();
        % Structure of each neural network
        network_layers;
        % Desired sample rate of the audio
        desired_fs = 16000;
        
    end
    
    methods (Access = public)
        
        % Constructor
        function obj = SpeechSeparator()
            
            % Check to see if speech_filter can be loaded
            try
                obj.bandpass_filter = load('speech_filter.mat').speech_filter;
            catch
                assert(0, 'Can not find speech_filter.mat. Is it in this directory?');
            end
            
        end
        
        % Audio vectors is a list of the audio vectors said by the speaker,
        % fs is the sample rate of all the clips (NEEDS TO BE THE SAME),
        % speaker_name is the name of the speaker
        function addNewSpeaker(obj, audio_vectors, fs, speaker_name)
            
            % Check to see if the speaker is already in the system, and if
            % so do nothing. This will need to be addressed later
            for names = 1 : obj.speech_data_size
                if (strcmp(obj.speech_data(names).name, speaker_name))
                    disp([speaker_name, ' already exists in the system.']);
                    return;
                end
            end
            
            % Create list of all audio data
            processed_audio_all = [];
            windows_all = [];
            speech_time = 0;
            
            % Loop through all the audio vectors
            for v = 1 : length(audio_vectors(:, 1))
                
                % Pre-process the audio clip
                audio = audio_vectors(v, :);
                [processed_audio, fs] = preProcessAudio(obj, audio, fs);
            
                % Break the audio into windows according to window_size
                windows = makeAudioWindows(obj, processed_audio, fs);
                
                % Append the new data to the list
                processed_audio_all = [processed_audio_all ; processed_audio];
                windows_all = [windows_all; windows];
                
                % Update the amount of speech time
                speech_time = speech_time + (length(processed_audio) / fs);
            end
            
            
            % Append data to global speaker struct
            obj.speech_data(obj.speech_data_size + 1).name = speaker_name;
            obj.speech_data(obj.speech_data_size + 1).original_audio = audio;
            obj.speech_data(obj.speech_data_size + 1).speech_time = speech_time;
            obj.speech_data(obj.speech_data_size + 1).fs = fs;
            obj.speech_data(obj.speech_data_size + 1).processed_audio = processed_audio_all;
            obj.speech_data(obj.speech_data_size + 1).windows = windows_all;
            obj.speech_data_size = obj.speech_data_size + 1;
            
            % Parition the data into training and test sets
            partitionData(obj);
            
        end
        
        % Begin training the neural networks if enough users exist in the
        % system
        function trainNetworks(obj)
            % Check if the training and testing datasets exist, and if so
            % begin training the neural networks
            if (obj.dataset_size == 0)
                disp('Not enough users to train filter masks.');
                return;
            end
            
            % Define the structure of each neural network
            numNodes = 4000;
            obj.network_layers = [
                
                featureInputLayer((obj.desired_fs / obj.window_size))
                
                fullyConnectedLayer(numNodes)
                tanhLayer
                batchNormalizationLayer
                dropoutLayer(0.75)
                
                fullyConnectedLayer(numNodes)
                tanhLayer
                batchNormalizationLayer
                dropoutLayer(0.75)
                
                fullyConnectedLayer((obj.desired_fs / obj.window_size))
                tanhLayer
                
                regressionLayer];
            
            % Train the networks with the given parameters
            disp('Training the network(s).');
            % Delete the previous filter masks
            obj.filter_masks = struct();
            for s = 1 : obj.dataset_size
                
                training_options = trainingOptions('adam',...
                    'Plots', 'training-progress',...
                    'MiniBatchSize', 64,...
                    'Shuffle', 'every-epoch',...
                    'ExecutionEnvironment', 'cpu',...
                    'MaxEpochs', 20,...
                    'ValidationData', obj.testing_data(s).validation_data,...
                    'ValidationFrequency', 50,...
                    'GradientThreshold', 1,...
                    'InitialLearnRate',0.001,...
                    'LearnRateSchedule','piecewise',...
                    'LearnRateDropPeriod',1,...
                    'LearnRateDropFactor',0.5);
                current_net = trainNetwork(obj.training_data(s).data, obj.network_layers, training_options);
                obj.filter_masks(s).net = current_net;
                obj.filter_masks(s).speaker = obj.training_data(s).name;
            end
        end
        
        function separated_audio = separateSpeech(obj, audio, fs, known_speaker)
            
            % Determine which model to use based on the speaker
            for s = 1 : length(obj.filter_masks)
                
                % Use the NN for the speaker
                if (strcmp(known_speaker, obj.filter_masks(s).speaker))
                    net = obj.filter_masks(s).net;
                    break
                end
                
            end
            
            % If the speaker is not in the system, return
            if (~(exist('net', 'var')))
                disp([known_speaker, ' has not been added to the system yet, and therefore has no trained model for speech separation.']);
                return
            end
            
            % Pre-process the audio vector
            [processed_audio, new_fs] = preProcessAudio(obj, audio, fs);
            
            % Prepare the data to be run through the network by creating
            % windows and calculating FFTs
            windows = makeAudioWindows(obj, processed_audio, new_fs);
            fft_windows = zeros(length(windows(:, 1)), length(windows(1, :)));
            output_windows = fft_windows;
            for w = 1 : length(windows(:, 1))
                temp_fft = fft(windows(w, :));
                temp_fft = temp_fft(1 : floor(length(temp_fft) / 2));
                fft_windows(w, :) = temp_fft;
            end
            
            % Run the ffts through the network
            for w = 1 : length(fft_windows(:, 1))
                output_windows(w, :) = ifft(net.predict(fft_windows(w, :)));
            end
            
        end
    end
    
    methods (Access = private)
        
        % Apply the audio pre-processing to the specified entry. This
        % function is copied from the SpeechProcessing class.
        function [speech_vector, new_fs, speech_indices, norm_audio] = preProcessAudio(obj, audio, fs)
            
            disp('Pre-processing the audio clip(s).');
            
            % Make sure audio is sampled at the correct frequency, and if
            % not resample it
            if (fs > obj.desired_fs)
                audio = resampleAudio(obj, audio, fs, obj.desired_fs);
                fs = 16000;
            end
            
            % If the audio is not a column vector, reshape it
            if (~(length(audio(:, 1)) == 1))
                audio = audio.';
            end
                        
            % Apply bandpass filter
            audio = obj.bandpass_filter.filter(audio);
            
            % Normalize audio
            data = audio / max(audio);
            
            % Cut out silences
            speechIdx = detectSpeech(data.', fs);
            speech_vector = [];
            for i = 1 : length(speechIdx(:, 1))
                speech_vector = [speech_vector, data(speechIdx(i, 1) : speechIdx(i, 2))];
            end
            
            if (nargout > 1)
                new_fs = fs;
            end
            
            if (nargout > 2)
                speech_indices = speechIdx;
            end
            
            if (nargout > 3)
                norm_audio = data;
            end      
        end
        
        % Break the desired audio stream into windows of given size and
        % overlap
        function windows = makeAudioWindows(obj, audio, fs)
            
            % Convert window size and overlamp to sample domain
            window_size_samples = obj.window_size * fs;
            window_overlap_samples = obj.window_overlap * fs;
            
            % Pre-allocate the windows matrix
            window_delta_samples = window_size_samples - window_overlap_samples;
            window_amount = ceil(length(audio) / window_delta_samples);
            
            % Check to make sure the clip is long enough to break into
            % windows. If not, return empty array.
            if (window_amount == 0)
                windows = [];
                return;
            end
            
            windows = zeros(window_amount, window_size_samples);
            
            % Check to see if there is enough audio for a single window. If
            % not, pad the end with zeroes
            if (length(audio) < window_size_samples)
                windows(1, 1 : length(audio)) = audio(1 : end);
            % If not, assign the values of the windows matrix
            else
                windows(1, :) = audio(1 : window_size_samples); 
            end
            for row = 2 : window_amount
                % Check to see if the window is too large
                if ((window_size_samples + (window_delta_samples * (row - 1)) - 1) > length(audio))
                    windows(row, 1 : length(audio(window_delta_samples * (row - 1) : end))) = audio(window_delta_samples * (row - 1) : end);
                    continue;
                end
                windows(row, :) = audio(window_delta_samples * (row - 1) : window_size_samples + (window_delta_samples * (row - 1)) - 1);
            end
            
        end
        
        % Re-sample the audio vector
        function resampled_audio = resampleAudio(~, audio, fs, desired_rate)
            
            % Ensure the desired rate is lower than the original
            assert(fs > desired_rate, 'New sample rate must be lower than the original.');
            
            % Calculate numerator and denominator to achieve desired sample
            % rate
            ratio = desired_rate / fs;
            [num, denom] = rat(ratio);
            
            % Resample audio
            resampled_audio = resample(audio, num, denom);
        end
        
        function partitionData(obj)
            
            % Ensure there are at least two speakers in the dataset
            if (obj.speech_data_size < 2)
                disp('Need more than 1 speaker to begin creating datasets.');
                return;
            end
            
            disp('Partitioning the data into training and testing sets.');
            
            % Loop through all the speaker data and create a global
            % training pool of overlapping speech. The size of this pool
            % will grow factorially, so be wary of adding many speakers to
            % the system.
            
            % Erase previous data
            obj.training_data = struct();
            obj.testing_data = struct();
            obj.dataset_size = 0;
            
            % Get the original speaker whose speech the other windows will
            % be added to
            for new_speaker = 1 : length(obj.speech_data)
                
                % Simplifies variable calls
                new_speaker_data = obj.speech_data(new_speaker);
                
                % Print what speaker is being calculated
                disp(['Calculating and partitioning data for ', new_speaker_data.name]);

                % Calculate the index on where to cutoff the training and
                % testing datasets
                training_index_cutoff = floor(length(new_speaker_data.windows(:, 1)) * obj.training_percentage);
                testing_index_cutoff = training_index_cutoff + 1;

                % Create the matrices that will be used for making both training
                % and testing data sets
                new_training_data = new_speaker_data(1).windows(1 : training_index_cutoff, :);
                new_testing_data = new_speaker_data(1).windows(testing_index_cutoff : end - 1, :);
                
                % Create empty matrices that will store all of the
                % overlapping speech. This need to be pre-allocated because
                % of the volume of data that will be created
                training_overlapping_windows_amount = 0;
                testing_overlapping_windows_amount = 0;
                training_data_windows_amount = length(new_training_data(:, 1));
                testing_data_windows_amount = length(new_testing_data(:, 1));
                for other_speaker = 1 : length(obj.speech_data)
                    
                    % Don't include this speaker in the calculation
                    if (new_speaker == other_speaker)
                        continue;
                    end
                    
                    % Determine how many rows the matrices will have
                    training_overlapping_windows_amount = training_overlapping_windows_amount + (training_data_windows_amount * length(obj.speech_data(other_speaker).windows(:, 1)));
                    testing_overlapping_windows_amount = testing_overlapping_windows_amount + (testing_data_windows_amount * length(obj.speech_data(other_speaker).windows(:, 1)));
                end
                new_training_input = zeros(training_overlapping_windows_amount, (obj.desired_fs / obj.window_size));
                new_testing_input = zeros(testing_overlapping_windows_amount, (obj.desired_fs / obj.window_size));
                new_training_output = new_training_input;
                new_testing_output = new_testing_input;
                % Loop through the data AGAIN and use the speech from the
                % OTHER speakers and add it to this speaker. This will
                % create windows of overlapping speech
                current_training_idx = 1;
                current_testing_idx = 1;
                for other_speaker = 1 : length(obj.speech_data)
                    
                    % Don't add overlapping speech from the same speaker
                    if (new_speaker == other_speaker)
                        continue;
                    end
                    
                    % Grab the windows from the speaker
                    other_speaker_speech = obj.speech_data(other_speaker).windows;                    
                    
                    % Loop through the current speakers windows
                    for cs = 1 : training_data_windows_amount
                        
                        % Loop through the other speakers windows
                        for os = 1 : length(other_speaker_speech(:, 1))                                                        
                            % Add the speech together
                            overlapping_vector = other_speaker_speech(os, :) + new_training_data(cs, :);
                            % Compute the FFT for each of the speech vectors
                            target_speaker_fft = fft(new_training_data(cs, :));
                            mixed_speaker_fft = fft(overlapping_vector);
                            % Cut out the latter half of the vector since
                            % its an even func
                            target_speaker_fft = target_speaker_fft(1 : floor(length(target_speaker_fft) / 2));
                            mixed_speaker_fft = mixed_speaker_fft(1 : floor(length(mixed_speaker_fft) / 2));
                            % Separate out the real and imaginary parts of
                            % the fft and use them as separate inputs
                            target_speaker_fft = [real(target_speaker_fft); imag(target_speaker_fft)];
                            target_speaker_fft = target_speaker_fft(:)';
                            mixed_speaker_fft = [real(mixed_speaker_fft); imag(mixed_speaker_fft)];
                            mixed_speaker_fft = mixed_speaker_fft(:)';
                            % Assign the input as the zscores of the data
                            new_training_input(current_training_idx, :) = zscore(mixed_speaker_fft);
                            % Assign the output as the zscores of the data
                            new_training_output(current_training_idx, :) = zscore(target_speaker_fft);
                            % Iterate idx
                            current_training_idx = current_training_idx + 1;
                        end
                    end
                    
                    % Repeat the previous process but for the testing data
                    for cs = 1 : testing_data_windows_amount
                        
                        % Loop through the other speakers windows
                        for os = 1 : length(other_speaker_speech(:, 1))
                            
                            % Add the speech together
                            overlapping_vector = other_speaker_speech(os, :) + new_testing_data(cs, :);
                            % Compute the FFT for each of the speech vectors
                            target_speaker_fft = fft(new_testing_data(cs, :));
                            mixed_speaker_fft = fft(overlapping_vector);
                            % Cut out the latter half of the vector since
                            % its an even func
                            target_speaker_fft = target_speaker_fft(1 : floor(length(target_speaker_fft) / 2));
                            mixed_speaker_fft = mixed_speaker_fft(1 : floor(length(mixed_speaker_fft) / 2));
                            % Separate out the real and imaginary parts of
                            % the fft and use them as separate inputs
                            target_speaker_fft = [real(target_speaker_fft); imag(target_speaker_fft)];
                            target_speaker_fft = target_speaker_fft(:)';
                            mixed_speaker_fft = [real(mixed_speaker_fft); imag(mixed_speaker_fft)];
                            mixed_speaker_fft = mixed_speaker_fft(:)';
                            % Assign the input as the zscores of the data
                            new_testing_input(current_testing_idx, :) = zscore(mixed_speaker_fft);
                            % Assign the output as the zscores of the data
                            new_testing_output(current_testing_idx, :) = zscore(target_speaker_fft);
                            % Iterate idx
                            current_testing_idx = current_testing_idx + 1;
                        end
                    end
                end
                
                % Append the final matrices to their respective structs
                training_input_table = array2table(new_training_input);
                training_output_table = array2table(new_training_output);
                data = [training_input_table, training_output_table];                
                obj.training_data(obj.dataset_size + 1).data = data;
                obj.training_data(obj.dataset_size + 1).name = new_speaker_data.name;
                
                % Rename data to meet the network validation standard
                testing_input_table = array2table(new_testing_input);
                testing_output_table = array2table(new_testing_output);
                validation_data = [testing_input_table, testing_output_table];
                obj.testing_data(obj.dataset_size + 1).validation_data = validation_data;
                obj.testing_data(obj.dataset_size + 1).name = new_speaker_data.name;
                
                % Iterate dataset size
                obj.dataset_size = obj.dataset_size + 1;
            end
            disp('Training and validation data has been caluclated and partioned.');
        end
    end
end
