close all;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = fullfile(datafolder, 'experiments');
gridfile = fullfile(datafolder, 'grid_greenscreen.MOV');
moviefile = fullfile(datafolder, 'red_dark_greenscreen.MOV');
resfolder = fullfile(datafolder, 'results');
outfile = fullfile(datafolder, 'out_red_dark_greenscreen_space_all.mat');

ncorners = 1; % total ncorners in the scene
corner_idx = 1; % corners are saved as TL, BL, BR, TR (if more than 1)

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.inf_method = 'naive';
params.amat_method = 'interp';
params.endframe = params.endframe / 4;

outframes = doCornerRecon(params, moviefile);
figure; imagesc(outframes);