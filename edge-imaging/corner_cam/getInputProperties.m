function params = getInputProperties(moviefile)
% sets isvid, frame_rate, maxframes, navg in params

% if moviefile is a file, we assume it's video, and gridfile
% is also a video. if it is a directory, we assume gridfile
% is a single image.
switch exist(moviefile)
case 2 % is a normal video
    v = VideoReader(moviefile);
    params.isvid = 1;
    params.frame_rate = v.FrameRate;
    params.start = round(params.frame_rate * 2);
    params.navg = 1;
    params.step = 1;
    params.maxframe = v.Duration * v.FrameRate;

case 7 % is a directory
    % photos are listed as "photo_n_m" where n is
    % the frame number and m is the copy number of the frame

    % find largest frame listed
    files = dir(moviefile);
    maxframe = 1;
    maxcopy = 1;
    for i = 1:length(files)
        [~, name] = fileparts(files(i).name);
        split = strsplit(name, '_');
        if length(split) < 3
            continue; % ., .., or doesn't follow expected naming
        end
        maxframe = max(str2double(split(2)), maxframe);
        maxcopy = max(str2double(split(3)), maxcopy);
    end
    params.isvid = 0;
    params.start = 1; % no crop
    params.frame_rate = 1; % /shrug
    params.maxframe = maxframe;
    params.navg = maxcopy;
    params.step = 1;
    
otherwise
    % check if we're dealing with point grey images
    testname = sprintf('%s-%4.4d.pgm', moviefile, 1);
    if exist(testname, 'file') == 2
        % find the largest frame listed
        path = fileparts(testname);
        files = dir(path);
        maxframe = 0;
        for i = 1:length(files)
            [~, name] = fileparts(files(i).name);
            split = strsplit(name, '-');
            if length(split) < 5
                continue; % ., .., or doesn't follow expected naming
            end
            numext = split(end);
            maxframe = max(str2double(numext), maxframe);
        end
        params.isvid = -1; % point grey
        params.start = 1; % no cropping
        params.frame_rate = 30.3;
        params.maxframe = maxframe;
        params.navg = 1;
        params.step = 1;
    else
        error('movie file not accepted type, or does not exist');
    end
end
end
