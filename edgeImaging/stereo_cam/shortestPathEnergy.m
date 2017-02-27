
datapath = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20';
resfolder = fullfile(datapath, 'results');
track1_path = fullfile(resfolder, 'corner1_red12_tracks.png');
track2_path = fullfile(resfolder, 'corner2_red12_tracks.png');
vidname = 'outputdepth.avi';

% load the 2 trajectory images corresponding to the 2 180 degree corner cams

x2_1d = double(imread(track1_path))./255;
x1_1d = double(imread(track2_path))./255;

% median filter the trajectories 
for c=1:3
    x2_1d(:,:,c) = medfilt2(x2_1d(:,:,c),[5 5]);
    x1_1d(:,:,c) = medfilt2(x1_1d(:,:,c),[5 5]);
end

% add some random noise so we avoid dividing by 0
x2_1d = double(x2_1d) + 0.001*randn(size(x2_1d));
x1_1d = double(x1_1d) + 0.001*randn(size(x1_1d));

baseline = 1; % set the size of the baseline. This will just scale the depth
default = 1e-2; 
winsize = 20;
nsamples = size(x1_1d,2);

% loop over each 1D frame in time
for f = 1:size(x2_1d,1)
    % for every window in x1_1d, we find the best matching window in x2_1d
    energy = inf(nsamples-winsize);
    
    for i = 1:nsamples-winsize
        win1 = x1_1d(f,i:i+winsize,:);
        for j = 1:i % only look in matching one side
            win2 = x2_1d(f, j:j+winsize,:);
            energy(i,j) = sum(sum((win2-win1).^2)); % take the mse of the windows
        end
    end
    
    
    % display the two images for each time step and the energy between
    % their matches
    subplot(232); imagesc(x2_1d(f,:,:));
    subplot(234); imagesc(permute(x1_1d(f,:,:), [2 1 3]));
    subplot(235); imagesc(energy); title('energy between two reconstructions');
    
    % form graph and do shortest path alg
    
    movingweight = 1e-10;
    count = 2;
    s = [];
    t = [];
    weights = [];
    nnodes = size(energy,1);
    for p=1:nnodes
        p/nnodes
    for q=1:nnodes
    
    
        if (p==1)
    
            energynodes1(p,q) = count;
            s = [s 1];
            t = [t energynodes1(p,q)];
            weights = [weights 0];
            count = count + 1;
    
            energynodes2(p,q) = count;
            s = [s energynodes1(p,q)];
            t = [t energynodes2(p,q)];
            weights = [weights energy(p,q)];
            count = count + 1;
    
        elseif(p<nnodes)
    
            energynodes1(p,q) = count;
            s = [s energynodes2(p-1,q)];
            t = [t energynodes1(p,q)];
            weights = [weights 0];
            count = count + 1;
    
            energynodes2(p,q) = count;
            s = [s energynodes1(p,q)];
            t = [t energynodes2(p,q)];
            weights = [weights energy(p,q)];
            count = count + 1;
    
            if (q>1)
            s = [s energynodes2(p-1,q-1)];
            t = [t energynodes1(p,q)];
            weights = [weights movingweight];
            end
    
            if (q<nnodes)
                s = [s energynodes2(p-1,q+1)];
                t = [t energynodes1(p,q)];
                weights = [weights movingweight];
            end
    
        elseif(p==nnodes)
            s = [s energynodes2(p-1,q)];
            t = [t count];
            weights = [weights 0];
        end
    
    
    end
    end
    G = digraph(s,t, weights);
    p = plot(G,'EdgeLabel',G.Edges.Weight);
    
    [path, d] = shortestpath(G,1,max(t));
    
    highlight(p,path,'EdgeColor','g')
    
    
    blah = ismember(energynodes1(:), path(:));
    figure; imagesc(reshape(blah, size(energynodes1))); pause(0.1);
end
