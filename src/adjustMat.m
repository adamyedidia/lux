function adjustedMat = adjustMat(mat)
    adjustedMat = (mat - ones(size(mat))*mean2(mat))/std2(mat);
end