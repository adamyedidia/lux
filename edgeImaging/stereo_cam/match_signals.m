function [match_angle, energy, match_idx] = match_signals(x1_1d, x2_1d, door_angle2, winsize)
energy = zeros([length(x1_1d)-winsize, length(x2_1d)-winsize]);
for i = 1:length(x1_1d)-winsize
    win1 = x1_1d(i:i+winsize);
    for j = 1:length(x2_1d)-winsize
        win2 = x2_1d(j:j+winsize);
%         energy(i,j) = -sum((win2-mean(win2)).*(win1-mean(win1)))/(std(win1)*std(win2));
        energy(i,j) = sum((win2-win1).^2);
    end
end

match_angle = zeros([1, length(x1_1d)-winsize]);
match_idx = zeros(size(match_angle));
for i = 1:length(match_angle)
    row = energy(i,:);
    [val, idx] = min(row);
    nomin = row(row > val);
    val2 = min(nomin);
    if val/val2 > 0.5
        % too close together, assume we're not resolving to an actual object
        match_angle(i) = nan;
    else
        % take the angle of corner 2 of the best match
        match_angle(i) = door_angle2(idx);
        match_idx(i) = idx;
    end
end
end