addpath(genpath('../../../../edgeImaging/'));
rmpath(genpath('../../../../edgeImaging/corner_sim/'))

clear; close all;

datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb22';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');

vidtype = '.MP4';
gridfile = fullfile(expfolder, strcat('calibration2', vidtype));


names = {'red_randomwalking_paperfloor_loc2', 'red_circle_paperfloor_loc2', 'red_figure8_paperfloor_loc2', 'red_angleline_paperfloor_loc2', 'red_windowline_paperfloor', 'redblue_randomwalking_paperfloor', ' redline_bluecircle_paperfloor', 'redblue_circles_paperfloor', 'redblue_randomwalking_naturalights_paperfloor copy', 'redblue_randomwalking_naturalights_paperfloor', 'redblue_randomwalking_naturallights', 'redblue_randomwalking', 'redblue_circles', 'red_randomwalking_naturallights_loc2', 'red_randomwalking_loc2'};

for n = 1:length(names)

    name = names{n};
    moviefile = fullfile(expfolder, strcat(name, vidtype));

    ncorners = 4; % total ncorners in the scene
    corner_idx = [1, 2, 3, 4];

    theta_lims{1} = [pi/2, 0];  % top left
    theta_lims{2} = [pi, pi/2]; % bottom left
    theta_lims{3} = [0, pi/2];  % bottom right
    theta_lims{4} = [pi/2, pi]; % top right

    params = initParams(moviefile, gridfile, ncorners, corner_idx);
    params.sub_mean = 1;
    params.endframe = params.endframe;
    params.inf_method = 'spatial_smoothing';
    params.amat_method = 'interp';
    corners = params.corner;

    for i = 1:length(corner_idx)
        c = corner_idx(i);
        params.corner = corners(c,:);
        params.theta_lim = theta_lims{c};
        outframe = doCornerRecon(params, moviefile);
        %figure; imagesc(outframe);

        outframes{i} = outframe;
    end
 
    left_floor = [outframes{2}, outframes{1}];
    right_floor = [fliplr(outframes{4}), fliplr(outframes{3})];
 
    left_scene = [fliplr(outframes{1}), fliplr(outframes{2})];
    right_scene = [outframes{3}, outframes{4}];

    outfile = fullfile(resfolder, strcat('out_', name, '_', params.inf_method, '_', params.amat_method, '.mat'));
    save(outfile);

end
