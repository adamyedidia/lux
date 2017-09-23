function outfile = saveMeanImage(overwrite, isvid, moviefile,...
    start, endframe, default_mean, step)
% compute the mean image of the moviefile, starting from start,
% ending at endframe with option step

if nargin < 7
    step = 1;
end
if nargin < 6
    default_mean = 1;
end

% moviefile will either be a video or a directory with image frames
if default_mean
    mean_name = 'mean_img.mat';
else
    % record the range over which we take the mean
    mean_name = sprintf('mean_img_%d_%d.mat', start, endframe);
end

switch isvid
case 1
    split = strsplit(moviefile, '.'); % same name, with mat extension
    outfile = sprintf('%s_%s', char(split(1)), mean_name);
    vsrc = VideoReader(moviefile);
    framesize = size(read(vsrc, 1));
case 0
    outfile = fullfile(moviefile, mean_name);
    vsrc = moviefile;
    framesize = size(imread(fullfile(vsrc, 'photo_1_2.CR2')));
case -1
    outfile = sprintf('%s_%s', moviefile, mean_name);
    vsrc = moviefile;
    fname = sprintf('%s-%4.4d.pgm', vsrc, 1);
    framesize = size(demosaic(imread(fname), 'rggb'));
end
    
if overwrite || ~exist(outfile, 'file') 
    fprintf('Saving mean image in %s\n', outfile);

    count = 0;
    avg_img = zeros(framesize);
    for n = start:step:endframe
        avg_img = avg_img + getFrame(isvid, vsrc, n, 1, framesize);
        count = count + 1;
    end
    avg_img = avg_img / count;
    mean_pixel = mean(mean(avg_img, 1), 2);

    save(outfile, 'avg_img', 'mean_pixel', 'endframe');
end

% we already have a mean file, check if we have variance
varinfo = who('-file', outfile);
if ~ismember('variance', varinfo)
    % compute the variance
    fprintf('Computing the variance and putting in %s\n', outfile);
    load(outfile, 'avg_img');
    count = 0;
    variance = zeros(framesize);
    for n = start:step:endframe
        framen = getFrame(isvid, vsrc, n, 1, framesize);
        variance = variance + (framen - avg_img).^2;
        count = count + 1;
    end
    variance = variance / (count - 1);
    save(outfile, 'variance', '-append');
end
end
