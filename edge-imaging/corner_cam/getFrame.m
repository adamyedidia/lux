function frame = getFrame(isvid, vsrc, n, navg, framesize)
% frame n averaged over navg frames, from a video or directory of
% photos
switch isvid

case 1
    if navg == 1
        frame = double(read(vsrc, n));
        return;
    end
    frame = zeros(framesize);
    count = 0;
    for i = n:n+navg-1
        frame = frame + double(read(vsrc, i));
        count = count + 1;
    end
    frame = frame / count;
case 0
    if navg == 1
        fname = fullfile(vsrc, sprintf('photo_%d_2.png', n));
        frame = double(imread(fname));
        return;
    end
    frame = zeros(framesize);
    % assumes vsrc is a director to list of raw photos named
    % "photo_n_m.CR2", n is frame number, m is copy number
    count = 0;
    for i = 2:max(2, navg)
        fname = fullfile(vsrc, sprintf('photo_%d_%d.png', n, i));
        fprintf('reading %s\n', fname);
        frame = frame + double(imread(fname));
        count = count + 1;
    end
    frame = frame / count;
case -1
    if navg == 1
        fname = sprintf('%s-%4.4d.pgm', vsrc, n);
        frame = double(demosaic(imread(fname), 'rggb'));
        return;
    end
    frame = zeros(framesize);
    count = 0;
    for i = n:n+navg-1
        fname = sprintf('%s-%4.4d.pgm', vsrc, i);
        try
            frame = frame + double(demosaic(imread(fname), 'rggb'));
            count = count + 1;
        catch err
            fprintf('error computing mean at %s\n', fname);
        end
    end
    frame = frame / count;
end
