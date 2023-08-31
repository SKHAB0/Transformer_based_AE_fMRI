%Preprocessing job
cd .. 

present_path = pwd();

cluster = parcluster('local');
nWorkers = cluster.NumWorkers;
parpool('local', nWorkers);

% Subject list

filename = 'DEPRESSION/participants.tsv';
opts = detectImportOptions(filename, 'Delimiter', '\t', 'FileType', 'text');
subjects_tsv = readtable(filename, opts);
subjects = subjects_tsv.participant_id;

addpath(' /usr/local/src/spm12')
addpath('/usr/local/src/marsbar-0.45')

%Launch SPM12
spm fmri

user = getenv('USER'); % Will return the username for OSX operating systems; change to 'USERNAME' for Windows

parfor i = 1:length(subjects)

    subject = subjects{i};
    parentFolder = [present_path '/DEPRESSION/' subject];

    %Retrieve sessions
    allItems = dir(parentFolder);
    dirFlags = [allItems.isdir] & ~strcmp({allItems.name}, '.') & ~strcmp({allItems.name}, '..');
    subFolders = allItems(dirFlags);

    sessions = {subFolders.name};

    for j = 1:length(sessions)

        session = sessions{j};

        %Defining func and anat folders
        % folderPath_1 = [parentFolder, '/' session '/func/'];
        folderPath_2 = [parentFolder, '/' session '/anat/'];
        % folderPath_3 = [parentFolder, '/' session '/fmap/'];

        %Unzipping functionnal files
        % func_zip_files = dir(fullfile(folderPath_1, '*.gz'));

        % if isempty(func_zip_files)
        %     disp('No zipped files found in the folder.');
        % else
        %     % Loop through each zipped file and unzip its contents
        %     for i = 1:length(func_zip_files)
        %         zipFilePath = fullfile(folderPath_1, func_zip_files(i).name);
        %         gunzip(zipFilePath, folderPath_1);
        %         disp(['Unzipped file: ' func_zip_files(i).name]);
        %     end
        % end

        %Unzipping anatomical files
        % anat_zip_files = dir(fullfile(folderPath_2, '*.gz'));

        % if isempty(anat_zip_files)
        %     disp('No zipped files found in the folder.');
        % else
        %     % Loop through each zipped file and unzip its contents
        %     for k = 1:length(anat_zip_files)
        %         zipFilePath = fullfile(folderPath_2, anat_zip_files(k).name);
        %         gunzip(zipFilePath, folderPath_2);
        %         disp(['Unzipped file: ' anat_zip_files(k).name]);
        %     end

        % end

        %Unzipping fmap files
        % fmap_zip_files = dir(fullfile(folderPath_3, '*.gz'));

        % if isempty(fmap_zip_files)
        %     disp('No zipped files found in the folder.');
        % else
        %     % Loop through each zipped file and unzip its contents
        %     for i = 1:length(fmap_zip_files)
        %         zipFilePath = fullfile(folderPath_3, fmap_zip_files(i).name);
        %         gunzip(zipFilePath, folderPath_3);
        %         disp(['Unzipped file: ' fmap_zip_files(i).name]);
        %     end
        % end

        %diary(['logfile_' subject '.txt']);

        files = dir(fullfile(folderPath_2, '*T1w.nii')).name;

        matlabbatch = {};

        % Execute the code specified in the SPM GUI
        matlabbatch{1}.spm.spatial.preproc.channel.vols = {fullfile(folderPath_2, files)};
        matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
        matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
        matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
        matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'spm12/tpm/TPM.nii,1'};
        matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
        matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'spm12/tpm/TPM.nii,2'};
        matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
        matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'spm12/tpm/TPM.nii,3'};
        matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
        matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'spm12/tpm/TPM.nii,4'};
        matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
        matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'spm12/tpm/TPM.nii,5'};
        matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
        matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'spm12/tpm/TPM.nii,6'};
        matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
        matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
        matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
        matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
        matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
        matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
        matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
        matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];
        matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
        matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                                      NaN NaN NaN];

        matlabbatch{2}.cfg_basicio.file_dir.cfg_fileparts.files(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'channel', '()', {1}, '.', 'biascorr', '()', {':'}));
        matlabbatch{3}.spm.util.imcalc.input(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'channel', '()', {1}, '.', 'biascorr', '()', {':'}));
        matlabbatch{3}.spm.util.imcalc.input(2) = cfg_dep('Segment: c1 Images', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'tiss', '()', {1}, '.', 'c', '()', {':'}));
        matlabbatch{3}.spm.util.imcalc.input(3) = cfg_dep('Segment: c2 Images', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'tiss', '()', {2}, '.', 'c', '()', {':'}));
        matlabbatch{3}.spm.util.imcalc.input(4) = cfg_dep('Segment: c3 Images', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'tiss', '()', {3}, '.', 'c', '()', {':'}));
        matlabbatch{3}.spm.util.imcalc.output = 'skullStripped_biasCorrected_T1.nii';
        matlabbatch{3}.spm.util.imcalc.outdir(1) = cfg_dep('Get Pathnames: Directories', substruct('.', 'val', '{}', {2}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'p'));
        matlabbatch{3}.spm.util.imcalc.expression = 'i1.*(i2+i3+i4)';
        matlabbatch{3}.spm.util.imcalc.var = struct('name', {}, 'value', {});
        matlabbatch{3}.spm.util.imcalc.options.dmtx = 0;
        matlabbatch{3}.spm.util.imcalc.options.mask = -1;
        matlabbatch{3}.spm.util.imcalc.options.interp = 0;
        matlabbatch{3}.spm.util.imcalc.options.dtype = 64;
        matlabbatch{4}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'fordef', '()', {':'}));
        matlabbatch{4}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'channel', '()', {1}, '.', 'biascorr', '()', {':'}));
        matlabbatch{4}.spm.spatial.normalise.write.subj.resample(2) = cfg_dep('Image Calculator: ImCalc Computed Image: skullStripped_biasCorrected_T1.nii', substruct('.', 'val', '{}', {3}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'files'));
        matlabbatch{4}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                                  78 76 85];
        matlabbatch{4}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        matlabbatch{4}.spm.spatial.normalise.write.woptions.interp = 7;
        matlabbatch{4}.spm.spatial.normalise.write.woptions.prefix = 'w';

        %Retrieve the voxel size for the T1 image
        %Normally isotropic voxels are used for the T1 image so 1 1 1 is used

        %NEEDDDD THE VOXEL SIZE !!!! Write the Normalize fail

        files = dir(fullfile(folderPath_2, '*T1w.json')).name;
        jsonString = fileread(fullfile(folderPath_2, files));
        jsonData = jsondecode(jsonString);

        voxelWidth = jsonData.pdb_FOV / jsonData.pdb_MATRIXX * 10;
        voxelHeight = jsonData.pdb_FOV / jsonData.pdb_MATRIXY * 10;
        voxelDepth = jsonData.pdb_SLTHICK;

        voxelSize_anat = [voxelWidth, voxelHeight, voxelDepth];

        matlabbatch{4}.spm.spatial.normalise.write.woptions.vox = voxelSize_anat;
        matlabbatch{4}.spm.spatial.normalise.write.woptions.interp = 7;
        matlabbatch{4}.spm.spatial.normalise.write.woptions.prefix = 'w';

        spm_jobman('run', matlabbatch);
        %diary off;

    end

end
