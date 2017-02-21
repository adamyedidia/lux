% ncorners = 3;
% theta_lims{1} = [pi, pi/2];
% theta_lims{2} = [pi/2, 0];
% theta_lims{3} = [0, pi/2];
% direction{1} = 1; 
% direction{2} = 1; 
% direction{3} = -1; 


ncorners = 4;
theta_lims{1} = [pi/2, 0];
theta_lims{2} = [pi, pi/2];
theta_lims{3} = [0, pi/2];
theta_lims{4} = [pi/2, pi];
direction{1} = -1; 
direction{2} = -1; 
direction{3} = 1; 
direction{4} = 1; 


addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));
addpath(genpath('../corner_cam'));

close all;

%% Parameters and file locations

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb14';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/loc2_caligrid.MOV', expfolder);
backfile = sprintf('%s/loc2_dark.MOV', expfolder);
moviefile = sprintf('%s/loc2_dark.MOV', expfolder);
outfile = sprintf('%s/out_loc2_dark.MOV', resfolder);

params.inf_method = 'spatial_smoothing';
params.amat_method = 'interp';
params.nsamples = 50;
params.rs = 10:2:30;
params.outr = 50;
params.theta_lim = [pi/2, 0];

params.lambda = 15; % pixel noise
params.sigma = 0.4; % prior variance
params.alpha = 5e-3; % process noise

params.sub_background = 0;
params.sub_mean = 0;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 4;
params.start = 60*5;
params.step = 5;

params.minclip = 0;
params.maxclip = 2;

%% Reconstruction

caldata = saveCalData(0, moviefile, gridfile, backfile, params.start);
load(caldata, 'frame1', 'endframe');

params.endframe = endframe;

frame1 = blurDnClr(frame1, params.downlevs, params.filt);

params.framesize = size(frame1);

% get A matrix
amat_out = getAmat(params);
amat = amat_out{1};
if ~strcmp(params.amat_method, 'interp')   
    params.crop_idx = amat_out{2};
end

% spatial prior
bmat = eye(params.nsamples) - diag(ones([params.nsamples-1,1]), 1);

figure; imagesc(frame1(:,:,1)); hold on;
corners = zeros([ncorners, 2]);
wall_pts = zeros([ncorners, 2]);
for c = 1:ncorners
    title(sprintf('choose corner and wall point, %d', c));
    pts = ginput(2);
    corners(c,:) = pts(1,:);
    wall_pts = pts(2,:);
    plot(pts(:,1), pts(:,2), 'ro');
end

%%

for c=1:ncorners
    
    params.corner = corner(c,:);
    params.theta_lim = theta_lims{c}; 
    

    switch params.inf_method
        case 'no_smoothing'
            outframes{c} = noSmoothingRecon(moviefile, caldata, params, amat, bmat);
        case 'spatial_smoothing'
            outframes{c} = spatialSmoothingRecon(moviefile, caldata, params, amat, bmat);
        case 'kalman_filter'
            fmat = eye(params.nsamples);
            outframes{c} = kalmanFilterRecon(moviefile, caldata, params, amat, bmat, fmat);
        case 'kalman_smoothing'
            fmat = eye(params.nsamples);
            outframes{c} = kalmanSmoothingRecon(moviefile, caldata, params, amat, bmat, fmat);
        otherwise % default to naive corner cam
            outframes{c} = cornerRecon(moviefile, caldata, params, amat);
    end
    
%     if direction{c} == -1
%         outframes{c} = fliplr(outframes{c}); 
%     end
    
    figure; imagesc(outframes{c});
    
end

figure;imagesc([outframes{1} outframes{2}]);
figure;imagesc([outframes{3} outframes{4}]);

x1_1d = [outframes{1} outframes{2}]; 
x2_1d = [outframes{3} outframes{4}];

%%

% writeReconVideo(outfile, outframes);

%%


baseline = 1; 
default = 1e-2;

figure;
for f = 1:207

nsamples = size(x1_1d,2); 

% estimate depth
% need to flip one of them; reverse orientations
%angles1 = angles1(end:-1:1);

% for every window in x1_1d, we find the best matching window in x2_1d
winsize = nsamples;
energy = zeros(nsamples-winsize);


for c=1:3
blah(:,:,c) = conv2(x1_1d(:,:,c), fliplr(x2_1d(:,:,c)), 'same'); 
end


dist2 = sqrt(blah(:,:,1).^2 + blah(:,:,2).^2 + blah(:,:,3).^2);
dist2row = dist2(floor(size(dist2,1)/2), :);
[val, idx] = max(dist2row);
if(idx-nsamples/2 <0)
    
    blah2 = [zeros(size(x1_1d,1), abs(idx - nsamples/2),3) x2_1d];
    blah1 = [x1_1d zeros(size(x2_1d,1), abs(idx - nsamples/2),3)];
    
else
    blah1 = [zeros(size(x1_1d,1), abs(idx - nsamples/2),3) x1_1d];
    blah2 = [x2_1d zeros(size(x2_1d,1), abs(idx - nsamples/2),3)];
    
end
imshowpair(blah1,blah2)


for i = 1:nsamples-winsize
    win1 = x1_1d(:,i:i+winsize,:);
    for j = 1:nsamples-winsize
        win2 = x2_1d(:, j:j+winsize,:);
%         energy(i,j) = -sum((win2-mean(win2)).*(win1-mean(win1)))/(std(win1)*std(win2));
        energy(i,j) = sum(sum(sum((win2-win1).^2)));
    end
end

for i = 1:nsamples-winsize
    win1 = x1_1d(f,i:i+winsize,:);
    for j = 1:nsamples-winsize
        win2 = x2_1d(f, j:j+winsize,:);
%         energy(i,j) = -sum((win2-mean(win2)).*(win1-mean(win1)))/(std(win1)*std(win2));
        energy(i,j) = sum(sum((win2-win1).^2));
    end
end

subplot(121); imagesc(energy); title('energy between two reconstructions');

% the corresponding angles from the door corners, for each point
% only positive angles for this calculation
angles1 = linspace(0,pi,nsamples); 
angles2 = linspace(0,pi,nsamples); 
door_angle1 = abs(angles1(1:nsamples-winsize));
depths = zeros(size(door_angle1));


for i = 1:length(depths)
    row = energy(i,:);
    [val, idx] = min(row);
    secondbest = row(row > val);
    val2 = min(secondbest);
    if val/val2 > 1.0%0.8
        % too close together, assume we're not resolving to an actual object
        depths(i) = default;
    else
        % take the angle of corner 2 of the best match
        hold on; plot(idx,i, 'r*'); 
        a2 = angles2(idx);
        a1 = pi - door_angle1(i); % katie changed this
        depths(i) = cot(a1) + cot(a2);
    end
end

depths = baseline ./ depths;
locs1 = cot(door_angle1) .* depths - 1;
angle_ends = abs(angles1(1+winsize:nsamples));
locs2 = cot(angle_ends) .* depths - 1;


subplot(122);
x = locs2(depths < baseline./default); 
y = depths(depths < baseline./default); 
scatter(x,y,100*ones(size(x)), squeeze(x1_1d(f, depths < baseline./default, :)), 'filled'); 
xlim([-3, 3]);
ylim([-12, 12]);
title('preliminary depth estimation of each point');

pause(0.1);
end

