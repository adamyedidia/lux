
function grFiglistClick ()
global figlist;

fig         = 1+get(figlist.list,'Value');
currentList = get(figlist.list, 'String');

if (currentList(fig-1,7) == double(':'))
  visible = strcmp(get(fig,'Visible'),'on');
  grSetVisible(fig,~visible);
  set(0,'CurrentFigure',fig);
end;
