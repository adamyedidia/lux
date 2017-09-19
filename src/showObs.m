function showObs(dv,dh)
    load('/Users/adamyedidia/walls/src/pole_images/monitor_lines/b_dark/arr.mat')
    obsPlane = arr;
    obsPlaneSize = length(obsPlane);


    scene = getMultiBarScene(obsPlaneSize);
    
    transferMatrix = getTransferMatrixFromLocation(dv,dh,0.02,[obsPlaneSize obsPlaneSize]);
    
    tentativeObsPlane = transferMatrix * scene;
    adjustedTentObsPlane = adjustMat(tentativeObsPlane);

    colorbar;
  %  imagesc(adjustMat(obsPlane));
    imagesc(adjustedTentObsPlane);
end

function singleBarScene = getSingleBarScene(obsPlaneSize)
    firstSection = int16(15*obsPlaneSize/32);
    secondSection = int16(2*obsPlaneSize/32);
    thirdSection = obsPlaneSize - firstSection - secondSection;
    singleBarScene = [repmat([0 0 0],firstSection,1);repmat([1 1 1],secondSection,1);repmat([0 0 0],thirdSection,1)]; 
end

function multiBarScene = getMultiBarScene(obsPlaneSize)
    multiBarSceneUnstretched = [repmat([0 0 0],99,1);repmat([1 1 1],44,1);repmat([0 0 0],26,1);repmat([1 1 1],113,1);repmat([0 0 0],58,1);repmat([1 1 1],5,1);repmat([0 0 0],110,1);repmat([1 1 1],40,1);repmat([0 0 0],50,1);repmat([0 0 0],99,1)]; 
    multiBarScene = imresize(multiBarSceneUnstretched, [obsPlaneSize 3], 'nearest');
end

function legoScene = getLegoScene(obsPlaneSize)
%    legoSceneUnstretched = [repmat([0 0 0],6,1);repmat([0 0 1],8,1);repmat([1 1 0],8,1);repmat([1 0 0],8,1);repmat([0 0 0],6,1)];
    legoSceneUnstretched = [repmat([0 0 0],6,1);repmat([1 0 0],8,1);repmat([1 1 0],8,1);repmat([0 0 1],8,1);repmat([0 0 0],6,1)];    
    legoScene = imresize(legoSceneUnstretched, [obsPlaneSize 3], 'nearest');
end