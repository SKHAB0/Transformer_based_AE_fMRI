addpath('/usr/local/src/spm12')
addpath('/usr/local/src/marsbar-0.45')

%BOLD extraction job
present_path = pwd();

marsbar('on');

aal_file = 'AAL3/ROI_MNI_V7.nii';
filename = 'BILINGUALISM/participants.tsv';
opts = detectImportOptions(filename, 'Delimiter', '\t', 'FileType', 'text');
subjects_tsv = readtable(filename, opts);
subjects = subjects_tsv.participant_id;

for iterator = 1:length(subjects)
    try
        subject = subjects{iterator};
        parentFolder = [present_path '/BILINGUALISM/' subject];

        % Retrieve sessions
        allItems = dir(parentFolder);
        dirFlags = [allItems.isdir] & ~strcmp({allItems.name}, '.') & ~strcmp({allItems.name}, '..');
        subFolders = allItems(dirFlags);

        data_dir = ['BILINGUALISM/' subject '/func'];
        temp_dir = ['BILINGUALISM/' subject '/temporary_folder'];

        % Create temporary directory to store ROI images
        mkdir(temp_dir);
        fun_imgs = spm_select('FPList', data_dir, ['^swau.*' '.*\.nii$']);

        num_rois = 116; % for the AAL atlas
        all_time_series = cell(1, num_rois);

        % Loop over each region in the AAL atlas
        for region_num = 1:116 % there are 116 regions in the AAL atlas
            try
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
            catch region_err
                fprintf('Error processing region %d: %s\n', region_num, region_err.message);
                continue; % Continue with the next region if an error occurs
            end
        end

        rmdir(temp_dir, 's');
        % Save the time series to a file
        save(['BILINGUALISM/' subject '/BOLD_time_series.mat'], 'all_time_series');
    catch subject_err
        fprintf('Error processing subject %s: %s\n', subject, subject_err.message);
        continue; % Continue with the next subject if an error occurs
    end
end
