function minFlowStereoMatching()
%load('outframes.mat');

vidname = '/Users/klbouman/Downloads/outputdepth.avi';

% load the 2 trajectory images corresponding to the 2 180 degree corner cams
x2_1d = double(imread('/Users/klbouman/Downloads/corner1_red12_tracks.png'))./255;
x1_1d = double(imread('/Users/klbouman/Downloads/corner2_red12_tracks.png'))./255;

% median filter the trajectories 
for c=1:3
    x2_1d(:,:,c) = medfilt2(x2_1d(:,:,c),[5 5]);
    x1_1d(:,:,c) = medfilt2(x1_1d(:,:,c),[5 5]);
end

% add some random noise so we avoid dividing by 0
x2_1d = double(x2_1d) + 0.001*randn(size(x2_1d));
x1_1d = double(x1_1d) + 0.001*randn(size(x1_1d));

% write out a video showing the depth estimation and the min flow 
vout = VideoWriter(vidname);
vout.FrameRate = 10;
open(vout);

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
    
    
    %% form graph and do shortest path alg
    
    % movingweight = 1e-10;
    % count = 2;
    % s = [];
    % t = [];
    % weights = [];
    % nnodes = size(energy,1);
    % for p=1:nnodes
    %     p/nnodes
    % for q=1:nnodes
    %
    %
    %     if (p==1)
    %
    %         energynodes1(p,q) = count;
    %         s = [s 1];
    %         t = [t energynodes1(p,q)];
    %         weights = [weights 0];
    %         count = count + 1;
    %
    %         energynodes2(p,q) = count;
    %         s = [s energynodes1(p,q)];
    %         t = [t energynodes2(p,q)];
    %         weights = [weights energy(p,q)];
    %         count = count + 1;
    %
    %     elseif(p<nnodes)
    %
    %         energynodes1(p,q) = count;
    %         s = [s energynodes2(p-1,q)];
    %         t = [t energynodes1(p,q)];
    %         weights = [weights 0];
    %         count = count + 1;
    %
    %         energynodes2(p,q) = count;
    %         s = [s energynodes1(p,q)];
    %         t = [t energynodes2(p,q)];
    %         weights = [weights energy(p,q)];
    %         count = count + 1;
    %
    %         if (q>1)
    %         s = [s energynodes2(p-1,q-1)];
    %         t = [t energynodes1(p,q)];
    %         weights = [weights movingweight];
    %         end
    %
    %         if (q<nnodes)
    %             s = [s energynodes2(p-1,q+1)];
    %             t = [t energynodes1(p,q)];
    %             weights = [weights movingweight];
    %         end
    %
    %     elseif(p==nnodes)
    %         s = [s energynodes2(p-1,q)];
    %         t = [t count];
    %         weights = [weights 0];
    %     end
    %
    %
    % end
    % end
    % G = digraph(s,t, weights);
    % p = plot(G,'EdgeLabel',G.Edges.Weight);
    %
    % [path, d] = shortestpath(G,1,max(t));
    %
    % highlight(p,path,'EdgeColor','g')
    %
    %
    % blah = ismember(energynodes1(:), path(:));
    % figure; imagesc(reshape(blah, size(energynodes1)));
    
    
    %% seam carving - find the minimum path through the data
    
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
        tmp = Pot(ii,max(pix(ii+1)-1,1):min(pix(ii+1)+1,end));
        [mn,gg]=min(tmp);
        pix(ii)=gg+pix(ii+1)-1-(pix(ii+1)>1);
        im(ii,pix(ii),:)=bitand(ii,1);
    end
    
    %%
    
    
    % display the best path found through the energy
    subplot(235);hold on; plot(pix,1:size(energy,1), 'r');
    
    
    %define angles
    angles1 = linspace(0,pi,nsamples);
    angles2 = linspace(0,pi,nsamples);
    door_angle1 = abs(angles1(1:nsamples-winsize));

    % compute the matching angles
    a2 = angles2(pix);
    a1 = pi - door_angle1; % katie changed this
    depths = cot(a1) + cot(a2);
    
    % calculate the depths
    depths = baseline ./ depths;
    locs1 = cot(door_angle1) .* depths - 1;
    angle_ends = abs(angles1(1+winsize:nsamples));
    locs2 = cot(angle_ends) .* depths - 1;
    
    % plot the trajectories where each circle has the color of one of the
    % frames
    subplot(133);
    x = locs2(depths < baseline./default);
    y = depths(depths < baseline./default);
    scatter(x,y,100*ones(size(x)), squeeze(x1_1d(f, depths < baseline./default, :)), 'filled');
    xlim([-10, 10]);
    ylim([0, 3]);
    title('preliminary depth estimation of each point');
    
    
    % write out the frame
    framepic = getframe(gcf);
    writeVideo(vout, framepic.cdata);
    drawnow; pause(0.01);
    
end

% close the video
close(vout);