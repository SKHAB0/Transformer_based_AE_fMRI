addpath(' /usr/local/src/spm12')
addpath('/usr/local/src/marsbar-0.45')

%BOLD extraction job
present_path = pwd();

% cluster = parcluster('local');
% nWorkers = cluster.NumWorkers;
% parpool('local', nWorkers);
marsbar('on');

aal_file = 'AAL3/ROI_MNI_V7.nii';
filename = 'DEPRESSION/participants.tsv';
opts = detectImportOptions(filename, 'Delimiter', '\t', 'FileType', 'text');
subjects_tsv = readtable(filename, opts);
subjects = subjects_tsv.participant_id;

for iterator = 1:length(subjects)
    subject = subjects{iterator};
    parentFolder = [present_path '/DEPRESSION/' subject];

    %Retrieve sessions
    allItems = dir(parentFolder);
    dirFlags = [allItems.isdir] & ~strcmp({allItems.name}, '.') & ~strcmp({allItems.name}, '..');
    subFolders = allItems(dirFlags);

    sessions = {subFolders.name};

    for j = 1:length(sessions)
        session = sessions{j};

        for echo = 1:4
            n_echo = num2str(echo);

            %if exist([parentFolder 'BOLD_time_series_echo-' n_echo '.mat'], 'file') ~= 2
            try

                data_dir = ['DEPRESSION/' subject '/' session '/func'];
                temp_dir = ['DEPRESSION/' subject '/' session '/temporary_folder'];

                %Create temporary directory to store ROI images
                mkdir(temp_dir);
                fun_imgs = spm_select('FPList', data_dir, ['^swau.*echo-' n_echo '.*\.nii$']);

                disp(['Subject ', subject, ' for session ' session ' for echo ' n_echo 'fun_images found'])

                num_rois = 166; % for the AAL3 atlas
                all_time_series = cell(1, num_rois);

                % Loop over each region in the AAL atlas
                for region_num = 1:num_rois % there are 116 regions in the AAL atlas

                    % Create a mask for the current region
                    roi_mask = (spm_read_vols(spm_vol(aal_file)) == region_num);

                    % Create a new image with the mask data
                    roi_img = spm_vol(aal_file);
                    roi_img.fname = fullfile(temp_dir, ['temp_ROI_' num2str(region_num) 'echo-' n_echo '.nii']);
                    spm_write_vol(roi_img, roi_mask);

                    % Convert the image to a MarsBaR ROI object
                    roi_obj = maroi_image(struct('vol', spm_vol(roi_img.fname), 'binarize', 1, ...
                        'descrip', ['AAL region ' num2str(region_num)]));

                    % Extract BOLD time series for the current ROI
                    [Y, roi_info] = getdata(roi_obj, fun_imgs);

                    % Store the time series (you can process it further if needed)
                    all_time_series{region_num} = Y;

                end

                disp(['Subject ', subject, ' for session ' session ' for echo ' n_echo 'time series computed'])

            catch ME % ME is an object that stores information about the error
                warning(['Error occurred during processing subject ', subject, ' for session ' session ': ', ME.message]);
                continue;
            end % end try

            rmdir(temp_dir, 's');

            % Save the time series to a file
            save(['DEPRESSION/' subject '/' session '/BOLD_time_series_echo-' n_echo '_166.mat'], 'all_time_series');

        end

    end

end

