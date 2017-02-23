
baseline = 1; 
default = 1e-2;
nsamples = size(x1_1d,2); 

figure;
for f = 1:207

    % for every window in x1_1d, we find the best matching window in x2_1d
    winsize = 10;
    energy = zeros(nsamples-winsize);

    for i = 1:nsamples-winsize
        win1 = x1_1d(f,i:i+winsize,:);
        for j = 1:nsamples-winsize
            win2 = x2_1d(f, j:j+winsize,:);
    %         energy(i,j) = -sum((win2-mean(win2)).*(win1-mean(win1)))/(std(win1)*std(win2));
            energy(i,j) = sum(sum((win2-win1).^2));
        end
    end

    subplot(121); imagesc(energy); title('energy between two reconstructions');

    % the corresponding angles from the door corners, for each point
    % only positive angles for this calculation
    angles1 = linspace(0,pi,nsamples); 
    angles2 = linspace(0,pi,nsamples); 
    door_angle1 = abs(angles1(1:nsamples-winsize));
    depths = zeros(size(door_angle1));


    for i = 1:length(depths)
        row = energy(i,:);
        [val, idx] = min(row);
        secondbest = row(row > val);
        val2 = min(secondbest);
        if val/val2 > 1.0%0.8
            % too close together, assume we're not resolving to an actual object
            depths(i) = default;
        else
            % take the angle of corner 2 of the best match
            hold on; plot(idx,i, 'r*'); 
            a2 = angles2(idx);
            a1 = pi - door_angle1(i); % katie changed this
            depths(i) = cot(a1) + cot(a2);
        end
    end

    depths = baseline ./ depths;
    locs1 = cot(door_angle1) .* depths - 1;
    angle_ends = abs(angles1(1+winsize:nsamples));
    locs2 = cot(angle_ends) .* depths - 1;


    subplot(122);
    x = locs2(depths < baseline./default); 
    y = depths(depths < baseline./default); 
    scatter(x,y,100*ones(size(x)), squeeze(x1_1d(f, depths < baseline./default, :)), 'filled'); 
    xlim([-3, 3]);
    ylim([-12, 12]);
    title('preliminary depth estimation of each point');

    pause(0.1);
end

