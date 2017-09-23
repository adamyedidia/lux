
function grOnCreate()
global figlist

if (figlist.currentFreeze>0) 
  delete(gcbo);
else
  set(0,'CurrentFigure',gcbo);
  grSetTitle(get(gcbo,'Name')); 
  set(gcbo,'DeleteFcn','grOnDelete');
  set(gcbo,'Doublebuffer','on');
end;


