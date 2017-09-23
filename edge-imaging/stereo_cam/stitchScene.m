function [floor_stitched, floor_angles, scene_stitched, scene_angles] = ...
    stitchScene(side1, side2, angle1, angle2)
% the floor is [side1, side2], the scene is the floor flipped

floor_angles = [angle1, angle2];
floor_stitched = [side1, side2];
[floor_stitched, floor_angles] = avgRepeatColumns(floor_stitched, floor_angles);

scene_stitched = fliplr(floor_stitched);
scene_angles = fliplr(floor_angles);

figure;
% subplot(211); imagesc(floor_angles, 1:size(floor_stitched,1), floor_stitched);
% subplot(212); imagesc(scene_angles, 1:size(scene_stitched,1), scene_stitched);
subplot(211); imagesc(floor_stitched); title('floor view');
subplot(212); imagesc(scene_stitched); title('scene view');
end