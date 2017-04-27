
clear; close all;

datapath = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20';
resfolder = fullfile(datapath, 'results');
corner1_path = fullfile(resfolder, 'corner1_red12_tracks.png');
corner2_path = fullfile(resfolder, 'corner2_red12_tracks.png');
vidname = 'outputdepth.avi';

% load the 2 trajectory images corresponding to the 2 180 degree corner cams
% corner 1 is left side, corner 2 is right side
x_left = fliplr(double(imread(corner1_path)))./255;
x_right = double(imread(corner2_path))./255;

% median filter the trajectories 
for c=1:3
    x_left(:,:,c) = medfilt2(x_left(:,:,c),[5 5]);
    x_right(:,:,c) = medfilt2(x_right(:,:,c),[5 5]);
end
figure; subplot(121); imagesc(x_left); subplot(122); imagesc(x_right);

% add some random noise so we avoid dividing by 0
x_right = double(x_right) + 0.001*randn(size(x_right));
x_left = double(x_left) + 0.001*randn(size(x_left));

% write out a video showing the depth estimation and the min flow 
vout = VideoWriter(vidname);
vout.FrameRate = 10;
open(vout);

baseline = 1; % set the size of the baseline. This will just scale the depth
default = 1e-2; 
winsize = 20;
nsamples = size(x_right,2);

% loop over each 1D frame in time
for f = 1:size(x_left,1)
    % for every window in x1_1d, we find the best matching window in x2_1d
    energy = inf(nsamples-winsize);
    
    for i = 1:nsamples-winsize
        win1 = x_left(f,i:i+winsize,:);
        for j = 1:i % angle from the left side must be greater than that from the right
            win2 = x_right(f, j:j+winsize,:);
            energy(i,j) = sum(sum((win2-win1).^2)); % take the mse of the windows
        end
    end
    
    
    % display the two images for each time step and the energy between
    % their matches
    subplot(232); imagesc(x_left(f,:,:));
    subplot(234); imagesc(permute(x_right(f,:,:), [2 1 3]));
    subplot(235); imagesc(energy); title('energy between two reconstructions');
       
    
    % seam carving - find the minimum path through the data
    
    % normalize the energy
    energynorm = energy;
    energynorm(isinf(energynorm)) = nan;
    energynorm = bsxfun(@rdivide, energynorm, nansum(energynorm,2));
    energynorm(isnan(energynorm)) = inf;
    
    im = energynorm;
    G = energynorm;
    
    %find shortest path in G
    Pot=G;
    for ii=2:size(Pot,1)
        pp=Pot(ii-1,:);
        ix=pp(1:end-1)<pp(2:end);
        pp([false ix])=pp(ix);
        ix=pp(2:end)<pp(1:end-1);
        pp(ix)=pp([false ix]);
        Pot(ii,:)=Pot(ii,:)+pp;
    end
    
    %Walk down hill
    pix=zeros(size(G,1),1);
    [mn,pix(end)]=min(Pot(end,:));
    pp=find(Pot(end,:)==mn);
    pix(end)=pp(ceil(rand*length(pp)));
    
    im(end,pix(end),:)=nan;
    for ii=size(G,1)-1:-1:1
        %[mn,gg]=min(Pot(ii,pix+(-1:1)));
        [mn, gg] = min(Pot(ii,max(pix(ii+1)-1,1):min(pix(ii+1)+1,size(Pot,2))));
        pix(ii)=gg+pix(ii+1)-1-(pix(ii+1)>1);
        im(ii,pix(ii),:)=bitand(ii,1);
    end
    
    % display the best path found through the energy
    subplot(235);hold on; plot(pix,1:size(energy,1), 'r');
    
    
    % define angles
    angles_right = linspace(0,pi,nsamples);
    angles_left = linspace(0,pi,nsamples);
    door_angle_right = abs(angles_right(1:nsamples-winsize));
    door_angle_left = pi - angles_left(1:nsamples-winsize);

    % compute the matching angles
    a_right = door_angle_right;
    a_left = door_angle_left(pix);
    depths = cot(a_right) + cot(a_left);
    
    % calculate the depths for each angular window
    depths = baseline ./ depths;
    locs1 = cot(door_angle_right) .* depths - baseline/2;
    angle_ends = abs(angles_right(1+winsize:nsamples));
    locs2 = cot(angle_ends) .* depths - baseline/2;
    
    % plot the trajectories where each circle has the color of one of the
    % frames
    subplot(133);
    x = locs2(depths < baseline./default);
    y = depths(depths < baseline./default);
    scatter(x,y,100*ones(size(x)), squeeze(x_right(f, depths < baseline./default, :)), 'filled');
    xlim([-10, 10]);
    ylim([-10, 10]);
    title('preliminary depth estimation of each point');
    
    
    % write out the frame
    framepic = getframe(gcf);
    writeVideo(vout, framepic.cdata);
    drawnow; pause(0.01);
    
end

% close the video
close(vout);