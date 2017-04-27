
function grSetTitle(arg1,arg2)
global figlist;

if (isstr(arg1))  fig = gcf;  title = arg1;
            else  fig = arg1; title = arg2;
end;

% -- Set the title in the window
% -----------------------------------

set(fig,'Name',title);

% -- Set the title in the figlist
% -----------------------------------

visible = strcmp(get(fig,'Visible'),'on');
str = sprintf(' %02d :%c: %s',fig.Number,utIf(visible,'#',' '),title);

grListSet(fig-1,str);

