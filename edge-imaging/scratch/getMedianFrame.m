function medimg = getMedianFrame(vidfile, downlevs, fps, start_t, end_t)
[dir, name] = fileparts(vidfile);
savedim = fullfile(dir, sprintf('%s_medimg.mat', name));
if exist(savedim, 'file')
    load(savedim, 'medimg');
    return;
end

vsrc = VideoReader(vidfile);
filter = binomialFilter(5);
test = blurDnClr(double(readFrame(vsrc)), downlevs, filter);
[nrows, ncols, nchans] = size(test);

colorhist = zeros([nrows, ncols, nchans, 256]);
[ii, jj, kk] = ndgrid(1:nrows, 1:ncols, 1:nchans);
times = start_t: 1/fps: end_t;
for i = 1:length(times)
    vsrc.CurrentTime = times(i);
    frame = uint8(blurDnClr(double(readFrame(vsrc)), downlevs, filter));
    % colorvalues are 0 to 255, index must be 1 to 256
    idx = sub2ind(size(colorhist), ii, jj, kk, frame+1);
    colorhist(idx) = colorhist(idx) + 1;
end

cumhist = cumsum(colorhist, 4);
isless = cumhist < length(times)/2;
medimg = sum(isless, 4) + 1;
save(savedim, 'medimg');
end