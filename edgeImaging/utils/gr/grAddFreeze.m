
function grAddFreeze(i)
global figlist;

nextFreeze = figlist.currentFreeze + i;
if ((nextFreeze >= 0) & (nextFreeze < 7))
  figlist.currentFreeze = nextFreeze;
end;