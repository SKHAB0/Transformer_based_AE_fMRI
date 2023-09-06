%Preprocessing job

%cd ..

%Unzipping all files for ease
%find BILINGUALISM -type f -name "*.gz" -exec gunzip -f {} \;

%Activating parallele computing
present_path = pwd();
cluster = parcluster('local');
nWorkers = cluster.NumWorkers;
parpool('local', nWorkers);

% Subject list retrieval
filename = 'BILINGUALISM/participants.tsv';
opts = detectImportOptions(filename, 'Delimiter', '\t', 'FileType', 'text');
subjects_tsv = readtable(filename, opts);
subjects = subjects_tsv.participant_id;

%Setting paths
addpath(' /usr/local/src/spm12')
addpath('/usr/local/src/marsbar-0.45')

%Launch SPM12
spm fmri

%for each subject
parfor iterator = 1:length(subjects)

    subject = subjects{iterator};
    parentFolder = [present_path '/BILINGUALISM/' subject];

    %Defining func and anat folders
    folderPath_1 = [parentFolder '/func/'];
    folderPath_2 = [parentFolder '/anat/'];
    folderPath_3 = [parentFolder '/fmap/'];

    if exist([folderPath_1 'swau' subject '_task-rest_run-01_bold.nii'], 'file') ~= 2

        %diary(['logfile_' subject '.txt']);

        filePattern = 'y_sub*T1w.nii';
        files = dir(fullfile(folderPath_2, filePattern)).name;

        % Execute the code specified in the SPM GUI

        %Retrieve func file parameters from json file
        jsonFilePath = [folderPath_1 subject '_task-rest_run-01_bold.json'];
        jsonString = fileread(jsonFilePath);
        jsonData = jsondecode(jsonString);
        SliceTiming = jsonData.SliceTiming;
        length_SliceTiming = length(SliceTiming);


        % Resulting voxel size
        voxelSize = [1.8, 1.8, 1.8] %[FOV / matrix_x, FOV / matrix_y, spacing_between_slices];

        %Initialize matlab batch

        matlabbatch = {};

        matlabbatch{1}.spm.spatial.realignunwarp.data.scans = {
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,1']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,3']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,2']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,4']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,5']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,6']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,7']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,8']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,9']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,10']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,11']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,12']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,13']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,14']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,15']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,16']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,17']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,18']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,19']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,20']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,21']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,22']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,23']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,24']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,25']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,26']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,27']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,28']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,29']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,30']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,31']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,32']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,33']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,34']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,35']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,36']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,37']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,38']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,39']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,40']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,41']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,42']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,43']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,44']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,45']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,46']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,47']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,48']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,49']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,50']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,51']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,52']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,53']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,54']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,55']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,56']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,57']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,58']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,59']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,60']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,61']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,62']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,63']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,64']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,65']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,66']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,67']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,68']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,69']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,70']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,71']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,72']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,73']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,74']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,75']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,76']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,77']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,78']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,79']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,80']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,81']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,82']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,83']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,84']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,85']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,86']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,87']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,88']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,89']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,90']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,91']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,92']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,93']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,94']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,95']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,96']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,97']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,98']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,99']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,100']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,101']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,102']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,103']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,104']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,105']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,106']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,107']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,108']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,109']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,110']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,111']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,112']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,113']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,114']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,115']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,116']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,117']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,118']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,119']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,120']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,121']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,122']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,123']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,124']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,125']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,126']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,127']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,128']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,129']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,130']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,131']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,132']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,133']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,134']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,135']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,136']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,137']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,138']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,139']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,140']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,141']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,142']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,143']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,144']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,145']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,146']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,147']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,148']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,149']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,150']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,151']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,152']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,153']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,154']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,155']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,156']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,157']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,158']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,159']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,160']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,161']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,162']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,163']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,164']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,165']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,166']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,167']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,168']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,169']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,170']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,171']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,172']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,173']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,174']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,175']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,176']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,177']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,178']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,179']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,180']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,181']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,182']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,183']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,184']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,185']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,186']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,187']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,188']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,189']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,190']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,191']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,192']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,193']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,194']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,195']
                                                               [folderPath_1 subject '_task-rest_run-01_bold.nii,196']
                                                               };
        %%
        matlabbatch{1}.spm.spatial.realignunwarp.data.pmscan = '';
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.quality = 0.9;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.sep = 4;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.rtm = 0;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.einterp = 2;
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
        matlabbatch{1}.spm.spatial.realignunwarp.eoptions.weight = '';
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.jm = 0;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.sot = [];
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.rem = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.noi = 5;
        matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.rinterp = 7;
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.mask = 1;
        matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';

        matlabbatch{2}.spm.temporal.st.scans{1}(1) = cfg_dep('Realign & Unwarp: Unwarped Images (Sess 1)', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'sess', '()', {1}, '.', 'uwrfiles'));
        matlabbatch{2}.spm.temporal.st.nslices = length_SliceTiming;
        matlabbatch{2}.spm.temporal.st.tr = jsonData.RepetitionTime;
        matlabbatch{2}.spm.temporal.st.ta = jsonData.RepetitionTime * (1 - 1 / length_SliceTiming);
        matlabbatch{2}.spm.temporal.st.so = SliceTiming;
        matlabbatch{2}.spm.temporal.st.refslice = jsonData.RepetitionTime * 1000/2;
        matlabbatch{2}.spm.temporal.st.prefix = 'a';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref = {[folderPath_2 '/skullStripped_biasCorrected_T1.nii,1']};
        matlabbatch{3}.spm.spatial.coreg.estimate.source(1) = cfg_dep('Realign & Unwarp: Unwarped Mean Image', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'meanuwr'));
        matlabbatch{3}.spm.spatial.coreg.estimate.other(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.', 'val', '{}', {2}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('()', {1}, '.', 'files'));
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
        matlabbatch{4}.spm.spatial.normalise.write.subj.def = {[folderPath_2 files]};
        matlabbatch{4}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.', 'val', '{}', {3}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'cfiles'));
        matlabbatch{4}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                                  78 76 85];
        matlabbatch{4}.spm.spatial.normalise.write.woptions.vox = voxelSize;
        matlabbatch{4}.spm.spatial.normalise.write.woptions.interp = 7;
        matlabbatch{4}.spm.spatial.normalise.write.woptions.prefix = 'w';
        matlabbatch{5}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.', 'val', '{}', {4}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('()', {1}, '.', 'files'));
        matlabbatch{5}.spm.spatial.smooth.fwhm = 2 * voxelSize;
        matlabbatch{5}.spm.spatial.smooth.dtype = 0;
        matlabbatch{5}.spm.spatial.smooth.im = 0;
        matlabbatch{5}.spm.spatial.smooth.prefix = 's';

        spm_jobman('run', matlabbatch);
        %diary off;

    end

end
