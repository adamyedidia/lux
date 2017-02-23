close all;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
resfolder = sprintf('%s/results', datafolder);
outfile = sprintf('%s/out_red_dark_greenscreen_space_all.mat', resfolder);

ncorners = 1; % total ncorners in the scene
corner_idx = 1; % corners are saved as TL, BL, BR, TR (if more than 1)

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.endframe = params.endframe / 3;

outframes = doCornerRecon(params, moviefile);
figure; imagesc(outframes);