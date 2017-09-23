% the pole has width w and is centered at (dv, dh)
% dv and dh must vary between -1 and 1, and (dv,dh,w) = (0,0,tiny) will
% give you the identity matrix
% also the pole is approximated as a flat thing
% outShape is a 2-entry array that tells you the shape of the transfer
% matrix you want
function transferMatrix = getTransferMatrixFromLocation(dv, dh, w, outShape)

    transferMatrix = ones(outShape);
    
    maxI = outShape(1);
    maxJ = outShape(2);
    
    for i = 1:maxI
        for j = 1:maxJ
            x = -(2*i-maxI)/maxI;
            y = (2*j-maxJ)/maxJ;
            
            transferMatrix(i,j) = 1 - getBandedness(x,y,dv,dh,w);
        end
    end     
    
    imagesc(transferMatrix);
end

function bandedness = getBandedness(x, y, dv, dh, w)
    if abs((1+dv)*y + (1-dv)*x - 2*dh) <= 2*w 
        bandedness = 1;
    else
        bandedness = 0;
    end
end