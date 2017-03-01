addpath(genpath('../../../../edgeImaging/'));
rmpath(genpath('../../../../edgeImaging/corner_sim/'))

clear; close all;

datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb22';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');

vidtype = '.MP4';
gridfile = fullfile(expfolder, strcat('calibration1', vidtype));


names = {'red_randomwalking', 'red_randomwalking2', 'red_circlewalking1', 'red_figure8walking1', 'red_circlewalking2', 'red_randomwalking_naturallights', 'red_circlewalking_naturalights', 'red_circlewalking_paperfloor', 'red_randomwalking_paperfloor', 'blue_circlewalking_paperfloor', 'blue_randomwalking_paperfloor', 'blue_randomwalking_paperfloor_naturallights', 'red_circlewalking_paperfloor_coveredlights'};

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
    params.endframe = params.endframe - params.start;
    params.inf_method = 'spatial_smoothing';
    params.amat_method = 'interp';
    corners = params.corner;
    
    outframes = cell(size(corner_idx));
    angles = cell(size(corner_idx));

    for i = 1:length(corner_idx)
        c = corner_idx(i);
        params.corner = corners(c,:);
        params.theta_lim = theta_lims{c};
        
        [outframe, angle] = doCornerRecon(params, moviefile);
        outframes{i} = outframe;
        angles{i} = reshape(angle, [1, length(angle)]); % make row vector
    end
    
    left_angles = [angles{2}, angles{1}];
    left_floor = [outframes{2}, outframes{1}];
    left_floor = avgRepeatColumns(left_floor, left_angles);

    right_angles = [fliplr(angles{4}), fliplr(angles{3})]; 
    right_floor = [fliplr(outframes{4}), fliplr(outframes{3})]; 
    right_floor = avgRepeatColumns(right_floor, right_angles);

    left_scene = fliplr(left_floor);
    right_scene = fliplr(right_floor);

    outfile = fullfile(resfolder, strcat('out_', name, '_', params.inf_method, '_', params.amat_method, '.mat'));
    save(outfile);
end
