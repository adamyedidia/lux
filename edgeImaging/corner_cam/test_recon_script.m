close all;

datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb22';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');

gridfile = sprintf('%s/calibration1.MP4', expfolder);
moviefile = sprintf('%s/blue_circlewalking_paperfloor.MP4', expfolder);
outfile = sprintf('%s/out_blue_circlewalking_paperfloor.MOV', resfolder);

ncorners = 4; % total ncorners in the scene
corner_idx = 1; % corners are saved as TL, BL, BR, TR (if more than 1)

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.inf_method = 'naive';
params.amat_method = 'interp';
params.endframe = params.endframe / 4;

outframes = doCornerRecon(params, moviefile);
figure; imagesc(outframes);