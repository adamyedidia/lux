% testing the stereo door corner camera
addpath(genpath('../utils/pyr'));
addpath(genpath('../corner_cam'));

close all; clear;

npx = 80;
width = npx/12;
floornpx = 60;
nsamples = 60;

% door is centered at origin
door_corner1 = [-1 0 0];
door_corner2 = [1 0 0];
door_width = abs(door_corner2(1) - door_corner1(1));
% we look at the shadow at the two corners of the door
xlocs1 = linspace(door_corner1(1), door_corner1(1)-1, floornpx);
xlocs2 = linspace(door_corner2(1), door_corner2(1)+1, floornpx);
ylocs = linspace(0, 1, floornpx);

% we have two other door corners
door_thickness = 0.2;
door_corner3 = [-1, -0.2, 0];
door_corner4 = [1, -0.2, 0];
xlocs3 = linspace(door_corner3(1), door_corner3(1)+1, floornpx+1);
xlocs3 = xlocs3(2:end);
xlocs4 = linspace(door_corner4(1), door_corner4(1)-1, floornpx+1);
xlocs4 = xlocs4(2:end);
ylocs2 = ylocs - 0.2;

% images behind the door
scenex = linspace(-2.5, 2.5, npx);
scenez = linspace(0, door_corner2(1), npx);
gt_depths = zeros(size(scenex));
imdepths = [-4, -7];
imstarts = [npx/10, 8*npx/10];
nims = length(imstarts);
imgs = zeros([npx, npx, nims]);

obs1 = zeros([floornpx*floornpx, 1]);
obs2 = zeros(size(obs1));
obs3 = zeros(size(obs1));
obs4 = zeros(size(obs1));
const3 = 0; % corners 3 and 4 see a constant image from the other side
const4 = 0;

figure;
for i = 1:nims
    start = imstarts(i);
    imgs(:,start:start+width,i) = abs(imdepths(i))/abs(max(imdepths));
    sceney = ones(size(scenex)) * imdepths(i);
    subplot(nims,1,i); imagesc(scenex, sceney, imgs(:,:,i));
    title(sprintf('depth %d', -imdepths(i)));
    gt_depths = gt_depths + abs(sign(imgs(1,:,i)) .* sceney);
    
    % get observations
    amat1 = cornerAmat(door_corner1, xlocs1, ylocs, scenex, sceney, 1);
    obs1 = obs1 + amat1 * reshape(imgs(:,:,i), [npx*npx,1]);
    
    amat2 = cornerAmat(door_corner2, xlocs2, ylocs, scenex, sceney, 1);
    obs2 = obs2 + amat2 * reshape(imgs(:,:,i), [npx*npx,1]);
    
    amat3 = cornerAmat(door_corner3, xlocs3, ylocs2, scenex, sceney, -1);
    obs3 = obs3 + amat3 * reshape(imgs(:,:,i), [npx*npx,1]);
    const3 = const3 + amat3(1,:) * reshape(imgs(:,:,i), [npx*npx,1]);
    
    amat4 = cornerAmat(door_corner4, xlocs4, ylocs2, scenex, sceney, -1);
    obs4 = obs4 + amat4 * reshape(imgs(:,:,i), [npx*npx,1]);
    const4 = const4 + amat4(1,:) * reshape(imgs(:,:,i), [npx*npx,1]);
end
sigma = 0.5; beta = 100;

obs1_noisy = obs1 + sigma * randn(size(obs1));
obs2_noisy = obs2 + sigma * randn(size(obs2));
obs3_noisy = obs3 + sigma * randn(size(obs3));
obs4_noisy = obs4 + sigma * randn(size(obs4));

% plotting the shadow observations
figure; 
subplot(411); 
imagesc(xlocs1, ylocs, reshape(obs1_noisy, [floornpx, floornpx]));
subplot(412);
imagesc(xlocs2, ylocs, reshape(obs2_noisy, [floornpx, floornpx]));
subplot(413);
imagesc(xlocs3, ylocs2, reshape(obs3_noisy, [floornpx, floornpx]));
subplot(414);
imagesc(xlocs4, ylocs2, reshape(obs4_noisy, [floornpx, floornpx]));

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);

% reconstructing a 1d picture from each door corner
thetas1 = [pi, pi/2];
tdir1 = sign(thetas1(2) - thetas1(1));
[amat, ~, ~] = allPixelAmat(door_corner1, floornpx, nsamples, thetas1);
angles1 = linspace(thetas1(1), thetas1(2), nsamples) + tdir1*pi;
x1_1d = (amat'*amat/sigma^2 + bmat'*bmat*beta)\(amat'*obs1_noisy/sigma^2);

thetas2 = [0, pi/2];
tdir2 = sign(thetas2(2) - thetas2(1));
[amat, ~, ~] = allPixelAmat(door_corner2, floornpx, nsamples, thetas2);
angles2 = linspace(thetas2(1), thetas2(2), nsamples) + tdir2*pi;
x2_1d = (amat'*amat/sigma^2 + bmat'*bmat*beta)\(amat'*obs2_noisy/sigma^2);

% these two corners see a constant image from the other side of the door
thetas3 = [pi/2, 0];
tdir3 = sign(thetas3(2) - thetas3(1));
[amat, ~, ~] = allPixelAmat(door_corner3, floornpx, nsamples, thetas3);
angles3 = linspace(thetas3(1), thetas3(2), nsamples) + tdir3*pi;
x3_1d = (amat'*amat/sigma^2 + bmat'*bmat*beta)\(amat'*obs3_noisy/sigma^2);

thetas4 = [pi/2, pi];
tdir4 = sign(thetas4(2) - thetas4(1));
[amat, ~, ~] = allPixelAmat(door_corner4, floornpx, nsamples, thetas4);
angles4 = linspace(thetas4(1), thetas4(2), nsamples) + tdir4*pi;
x4_1d = (amat'*amat/sigma^2 + bmat'*bmat*beta)\(amat'*obs4_noisy/sigma^2);


% x1_1d = (x1_1d - min(x1_1d))/(max(x1_1d) - min(x1_1d));
% x2_1d = (x2_1d - min(x2_1d))/(max(x2_1d) - min(x2_1d));
% x3_1d = (x3_1d - min(x3_1d))/(max(x3_1d) - min(x3_1d));
% x4_1d = (x4_1d - min(x4_1d))/(max(x4_1d) - min(x4_1d));


figure; 
subplot(411); imagesc(angles1, 1:npx/2, repmat(x1_1d', [npx/2, 1]));
subplot(412); imagesc(angles2, 1:npx/2, repmat(x2_1d', [npx/2, 1]));
subplot(413); imagesc(angles3, 1:npx/2, repmat(x3_1d', [npx/2, 1]));
subplot(414); imagesc(angles4, 1:npx/2, repmat(x4_1d', [npx/2, 1]));

% estimate depth
% need to flip one of them; reverse orientations
x1_1d = x1_1d(end:-1:1);
angles1 = angles1(end:-1:1);
x3_1d = x3_1d(end:-1:1);
angles3 = angles3(end:-1:1);

% the corresponding angles from the door corners, for each point
% only positive angles for this calculation
winsize = 3;
door_angle1 = abs(angles1(1:nsamples-winsize));
door_angle2 = angles2(1:nsamples-winsize) - tdir2*pi;
door_angle3 = 3*pi/2 - angles4(1:nsamples-winsize);
door_angle4 = 2*pi - angles3(1:nsamples-winsize);

% for every window in x1_1d, we find the best matching window in x2_1d
[match12, energy12] = match_signals(x1_1d, x2_1d, door_angle2, winsize);
[match14, energy14] = match_signals(x1_1d, x4_1d, door_angle4, winsize);
[match23, energy23] = match_signals(x2_1d, x3_1d, door_angle3, winsize);

figure; 
subplot(221); imagesc(energy12); title('energy between corners 1 and 2');
subplot(222); imagesc(energy14); title('energy between corners 1 and 4');
subplot(223); imagesc(energy23); title('energy between corners 2 and 3');

% for corners 1 and 2
depths12 = door_width ./ cot(door_angle1) + cot(match12);
locs12 = cot(door_angle1) .* depths12;

% for corners 1 and 4
depths14 = (door_width * tan(door_angle1) - door_thickness) ./ ...
    (tan(match14) - tan(door_angle1)) .* tan(match14) + door_thickness;
locs14 = cot(door_angle1) .* depths14;

% for corners 2 and 3
depths23 = (door_width .* tan(door_angle2) - door_thickness) ./ ...
    (tan(match23) - tan(door_angle2)) .* tan(match23) + door_thickness;
locs23 = door_corner2(1) - cot(door_angle2) .* depths23;

figure;
subplot(411);
plot(locs12(~isnan(depths12)), depths12(~isnan(depths12)), 'ro'); title('using corners 1 and 2');
subplot(412);
plot(locs14(~isnan(depths14)), depths14(~isnan(depths14)), 'ro'); title('using corners 1 and 4');
subplot(413);
plot(locs23(~isnan(depths23)), depths23(~isnan(depths23)), 'ro'); title('using corners 2 and 3');
subplot(414);
plot(scenex(gt_depths > 0), gt_depths(gt_depths > 0), 'bo'); title('ground truth depths');
