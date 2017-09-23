% testing the stereo door corner camera
addpath(genpath('../utils/pyr'));
addpath(genpath('../corner_cam'));
addpath(genpath('../stereo_cam'));

close all; clear;

npx = 80;
width = npx/10;
floornpx = 80;
nsamples = 80;

% door is centered at origin
door_corner1 = [-1 0 0];
door_corner2 = [1 0 0];
door_width = abs(door_corner2(1) - door_corner1(1));
% we look at the shadow at the two corners of the door
xlocs1 = linspace(door_corner1(1), door_corner1(1)-1, floornpx);
xlocs2 = linspace(door_corner2(1), door_corner2(1)+1, floornpx);
ylocs = linspace(0, 1, floornpx);

% images behind the door
imdepths = [-5, -7];
imstarts = [2*width, 7*width];
nims = length(imstarts);
imgs = zeros([npx, npx, nims]);
scenex = linspace(door_corner1(1), door_corner2(1), npx);
sceneys = zeros([nims, npx]);
scenez = linspace(0, door_corner2(1), npx);
gt_depths = zeros(size(scenex));

obs1 = zeros([floornpx*floornpx, 1]);
obs2 = zeros(size(obs1));
figure;
for i = 1:nims
    start = imstarts(i);
    imgs(:,start:start+width,i) = 1;
    sceneys(i,:) = ones(size(scenex)) * imdepths(i);
    subplot(nims,1,i); imagesc(imgs(:,:,i));
    axis off; title(sprintf('%d units depth', -imdepths(i)));
    gt_depths = gt_depths + abs(imgs(1,:,i) .* sceneys(i,:));
    
    % get observations
    amat1 = cornerAmat(xlocs1, ylocs, scenex, sceneys(i,:));
    obs1 = obs1 + amat1 * reshape(imgs(:,:,i), [npx*npx,1]);
    
    amat2 = cornerAmat(xlocs2, ylocs, scenex, sceneys(i,:));
    obs2 = obs2 + amat2 * reshape(imgs(:,:,i), [npx*npx,1]);
end
sigma = 0.5; beta = 100;

obs1_noisy = obs1 + sigma * randn(size(obs1));
obs2_noisy = obs2 + sigma * randn(size(obs1));

% plotting the shadow observations
figure; 
subplot(211); 
imagesc(xlocs1, ylocs, reshape(obs1_noisy, [floornpx, floornpx]));
subplot(212);
imagesc(xlocs2, ylocs, reshape(obs2_noisy, [floornpx, floornpx]));

% reconstructing a 1d picture from each door corner
thetas1 = [pi, pi/2];
tdir1 = sign(thetas1(2) - thetas1(1));
[amat1_1d, ~, ~] = allPixelAmat(door_corner1, floornpx, nsamples, thetas1);
angles1 = linspace(thetas1(1), thetas1(2), nsamples) + tdir1*pi;

thetas2 = [0, pi/2];
tdir2 = sign(thetas2(2) - thetas2(1));
[amat2_1d, ~, ~] = allPixelAmat(door_corner2, floornpx, nsamples, thetas2);
angles2 = linspace(thetas2(1), thetas2(2), nsamples) + tdir2*pi;

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);

x1_1d = (amat1_1d'*amat1_1d/sigma^2 + bmat'*bmat*beta)\(amat1_1d'*obs1_noisy/sigma^2);
x2_1d = (amat2_1d'*amat2_1d/sigma^2 + bmat'*bmat*beta)\(amat2_1d'*obs2_noisy/sigma^2);

% x1_1d(x1_1d<0) = 0;
% x1_1d(x1_1d>1) = 1;
% x2_1d(x2_1d<0) = 0;
% x2_1d(x2_1d>1) = 1;

x1_1d = (x1_1d - min(x1_1d))/(max(x1_1d) - min(x1_1d));
x2_1d = (x2_1d - min(x2_1d))/(max(x2_1d) - min(x2_1d));

figure; 
subplot(211); imagesc(angles1, 1:npx/2, repmat(x1_1d', [npx/2, 1]));
subplot(212); imagesc(angles2, 1:npx/2, repmat(x2_1d', [npx/2, 1]));

% estimate depth
% need to flip one of them; reverse orientations
x1_1d = x1_1d(end:-1:1);
angles1 = angles1(end:-1:1);

% for every window in x1_1d, we find the best matching window in x2_1d
winsize = 2;
energy = zeros(nsamples-winsize);

for i = 1:nsamples-winsize
    win1 = x1_1d(i:i+winsize);
    for j = 1:nsamples-winsize
        win2 = x2_1d(j:j+winsize);
%         energy(i,j) = -sum((win2-mean(win2)).*(win1-mean(win1)))/(std(win1)*std(win2));
        energy(i,j) = sum((win2-win1).^2);
    end
end

figure; imagesc(energy); title('energy between two reconstructions');

% the corresponding angles from the door corners, for each point
% only positive angles for this calculation
door_angle1 = abs(angles1(1:nsamples-winsize));
depths = zeros(size(door_angle1));

for i = 1:length(depths)
    row = energy(i,:);
    [val, idx] = min(row);
    secondbest = row(row > val);
    val2 = min(secondbest);
    if val/val2 > 0.6
        % too close together, assume we're not resolving to an actual object
        depths(i) = 1e-2;
    else
        % take the angle of corner 2 of the best match
        a2 = angles2(idx) - tdir2*pi;
        a1 = door_angle1(i);
        depths(i) = cot(a1) + cot(a2);
    end
end

depths = abs(door_corner1(1) - door_corner2(1)) ./ depths;
locs1 = cot(door_angle1) .* depths - 1;
angle_ends = abs(angles1(1+winsize:nsamples));
locs2 = cot(angle_ends) .* depths - 1;

% [minval, minidx] = min(energy, [], 2);
% door_angle2 = angles2(minidx) - tdir2*pi;
% for each point, compute depth
% depths = abs(door_corner1(1) - door_corner2(1))./(cot(door_angle1) + cot(door_angle2));

figure; subplot(211);
plot(locs1(depths < 200), depths(depths < 200), 'ro');
hold on;
plot(locs2(depths < 200), depths(depths < 200), 'bo');
xlim([-1, 1]);
title('preliminary depth estimation of each point');
subplot(212); plot(scenex(gt_depths > 0), gt_depths(gt_depths > 0), 'bo');
xlim([-1, 1]); title('ground truth depths');
