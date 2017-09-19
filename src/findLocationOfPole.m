function locationOfPole = findLocationOfPole()
    load('/Users/adamyedidia/walls/src/pole_images/monitor_lines/b_dark/arr.mat')
    obsPlane = arr;
    obsPlaneSize = length(obsPlane);
    scene = getMultiBarScene(obsPlaneSize);
    
    locationOfPole = searchOverLocations(scene, obsPlane);
    
%    dv = locationOfPole(1);
%    dh = locationOfPole(2);
    
%    transferMatrix = getTransferMatrixFromLocation(dv,dh,0.02,[sceneLength sceneLength]);
                        
%    tentativeObsPlane = transferMatrix * scene;
%    adjustedTentObsPlane = adjustMat(tentativeObsPlane);

 %   colorbar;
%    imagesc(tentativeObsPlane);

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

function locationOfPole = searchOverLocations(scene, realObsPlane)

    sceneLength = length(scene);
    sceneSize = size(scene);
    obsPlaneSize = size(realObsPlane);
    
    adjustedRealObsPlane = adjustMat(realObsPlane);    
    
    colorbar;
%    size(repmat(realObsPlane, [1 1 2]))
    imagesc(adjustedRealObsPlane);
%    imagesc(permute(repmat(adjustedRealObsPlane, [1 1 2]), [1 3 2]));
    pause(6);
        
    bestError = 1e10;
    
    for dv = -1:0.025:1
        for dh = -1:0.025:1
            transferMatrix = getTransferMatrixFromLocation(dv,dh,0.02,[sceneLength sceneLength]);

            tentativeObsPlane = transferMatrix * scene;
            adjustedTentObsPlane = adjustMat(tentativeObsPlane);
            
            colorbar;
%            adjustedTentObsPlane
            imagesc(adjustedTentObsPlane);
%            imagesc(permute(repmat(adjustedTentObsPlane, [1 1 2]), [1 3 2]));
            
            pause(0.05);

            error = norm(adjustedRealObsPlane-adjustedTentObsPlane);
                        
            [error dv dh]
            
            if ~(isnan(error)) && error < bestError
                bestError = error;
                bestDv = dv;
                bestDh = dh;
            end
        end
    end
                
    locationOfPole = [bestDv bestDh];
end
    