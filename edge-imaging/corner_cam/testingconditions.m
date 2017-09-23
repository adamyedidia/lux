close all;

% specify directories
datafolder = '~/Downloads/testing_conditions';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');
moviefile = sprintf('%s/testing', expfolder); 

%% GENERATE DATA

sigma = 1; % gaussian sensor noise - VARY THIS
ambient = 0; % ambient intensity - VARY THIS

params_init.framesize = [50 50 1]; 
params_init.corner = [1 1];
params_init.rs = 4:2:30; 
params_init.nsamples = 200; 
params_init.theta_lim = [pi/2 0]; 
params_init.outr = 50; 
params_init.amat_method = 'allpixels';
[Amat, angles, params] = getAmat(params_init); 

% load image - VARY THIS
img = imresize(double(imread('circles.png'))./255, [params_init.nsamples, params_init.nsamples]);  
img_1D = sum(img,1);
% img_1D = zeros(size(img_1D));
% img_1D(end/2:end) = 1;
x = [ambient; img_1D(:)]; 

% generate clean measurements
y_clean = Amat{1} * x(:); 

% reshape; 
sz = min(params.outr, sqrt(length(y_clean))); 
y_clean = reshape(y_clean, [sz sz]); 

% add noise
y_noise = y_clean + sigma*randn(size(y_clean)); 

%y_noise_pad = zeros(params_init.framesize); 
%y_noise_pad(params.corner(1)+1:params.corner(1)+params.outr, params.corner(1)+1:params.corner(1)+params.outr,1) = y_noise; 

% save out image (quantize it)
imwrite(uint8(y_noise), sprintf('%s/photo_1_2.png', moviefile)); 

% visualize
figure;
subplot(222); imagesc(img_1D); title('1D Input Image'); colorbar;  
subplot(221); imagesc(y_clean); title('Clean Corner Image'); 
subplot(223); imagesc(y_noise); title('Noisy Corner Image'); 


%% INFERENCE

params = params_init; 
params.inf_method = 'spatial_smoothing'; % reconstruction method - VARY THIS 
params.amat_method = 'allpix'; %reconstruction method - VARY THIS 
params.isvid = 0; 
params.start = 1; 
params.step = 1; 
params.endframe = 1; 
params.sub_mean = 1; 
params.smooth_up = 1; 
params.navg = 1;
params.downlevs = 0; 
params.filt = binomialFilter(5);

params.lambda = sigma^2; % pixel noise
params.sigma = 0.4; % prior variance
params.alpha = 5e-3; % process noise
params.eps = 1e-5;


params.corner_datafile = sprintf('%s/testing_walking_ncorners=1_downlevs=0.mat', expfolder); 
params.cal_datafile = sprintf('%s/calibrationgrid.mat', expfolder); 
params.mean_datafile = sprintf('%s/testing_mean_img.mat', expfolder); 


avg_img = zeros(size(y_noise)); 
corners = params.corner; 
save(params.corner_datafile, 'avg_img', 'corners'); 

mean_pixel = 0; 
endframe = 1; 
save(params.mean_datafile, 'avg_img', 'endframe', 'mean_pixel'); 

[ii, jj] = meshgrid(1:sz, 1:sz); 
iold = ii; 
jold = jj; 
save(params.cal_datafile, 'ii', 'jj', 'iold', 'jold'); 


outframes = doCornerRecon(params, moviefile);
subplot(224); imagesc(outframes(2:end), [0 1]); imagesc(outframes(2:end)); title('1D Output Image'); colorbar; 


