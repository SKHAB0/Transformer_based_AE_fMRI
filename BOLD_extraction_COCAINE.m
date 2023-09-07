addpath('/usr/local/src/spm12')
addpath('/usr/local/src/marsbar-0.45')

%BOLD extraction job
present_path = pwd();

marsbar('on');

aal_file = 'AAL3/ROI_MNI_V7.nii';
filename = 'COCAINE/participants.tsv';
opts = detectImportOptions(filename, 'Delimiter', '\t', 'FileType', 'text');
subjects_tsv = readtable(filename, opts);
subjects = subjects_tsv.participant_id;

for iterator = 1:length(subjects)
    subject = subjects{iterator};
    parentFolder = [present_path '/COCAINE/' subject];

    disp(['Starting preprocessing for subject: ' subject])

    % Retrieve sessions
    allItems = dir(parentFolder);
    dirFlags = [allItems.isdir] & ~strcmp({allItems.name}, '.') & ~strcmp({allItems.name}, '..');
    subFolders = allItems(dirFlags);

    data_dir = ['COCAINE/' subject '/func'];
    temp_dir = ['COCAINE/' subject '/temporary_folder'];

    try
        % Create temporary directory to store ROI images
        mkdir(temp_dir);
    catch
    end

    fun_imgs = spm_select('FPList', data_dir, ['^swau.*' '.*\.nii$']);

    num_rois = 166; % for the AAL atlas
    all_time_series = cell(1, num_rois);

    try
        % Loop over each region in the AAL atlas
        parfor region_num = 1:166 % there are 166 regions in the AAL atlas
            % Create a mask for the current region
            roi_mask = (spm_read_vols(spm_vol(aal_file)) == region_num);

            % Create a new image with the mask data
            roi_img = spm_vol(aal_file);
            roi_img.fname = fullfile(temp_dir, ['temp_ROI_' num2str(region_num) '.nii']);
            spm_write_vol(roi_img, roi_mask);

            % Convert the image to a MarsBaR ROI object
            roi_obj = maroi_image(struct('vol', spm_vol(roi_img.fname), 'binarize', 1, ...
                'descrip', ['AAL region ' num2str(region_num)]));

            % Extract BOLD time series for the current ROI
            [Y, roi_info] = getdata(roi_obj, fun_imgs);

            % Store the time series (you can process it further if needed)
            all_time_series{region_num} = Y;
        end

        rmdir(temp_dir, 's');
        % Save the time series to a file
        save(['COCAINE/' subject '/BOLD_time_series.mat'], 'all_time_series');

    catch subject_err
        fprintf('Error processing subject %s: %s\n', subject, subject_err.message);
        continue; % Continue with the next subject if an error occurs
    end
    disp(['Done preprocessing for subject: ' subject])
end
