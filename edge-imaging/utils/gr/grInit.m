
function grInit()
global figlist;

% ---------------------------------------------

if (~isempty(figlist) & ~figlist.deleted)
  currentList = get(figlist.list, 'String');
  set(1,'HandleVisibility','on');
  for fig = 1:(1+size(currentList,1))
     delete(fig);
  end;
end;

% ---------------------------------------------

winSize    = [328 217];
screenSize = get(0,'ScreenSize');
winPos     = [10 screenSize(4)-winSize(2)-63];

figlist.fig = figure(1);
set(figlist.fig,'Color',[0.8 0.8 0.8], ...
    'Colormap',gray(10), 'Menubar', 'none', ...
    'Position',[winPos(1) winPos(2) winSize(1) winSize(2)]);
set(figlist.fig,'ResizeFcn', 'grResize');

  
figlist.list = uicontrol('Parent',figlist.fig, ...
  'Units','points', ...
  'BackgroundColor',[1 1 1], ...
  'Style','listbox', ...
  'SelectionHighlight','off', ...
  'Callback','grOnClick', ...
  'Value',1);
  
figlist.uiText = uicontrol('Parent',figlist.fig, ...
  'Units','points', ...
  'BackgroundColor',[0.8 0.8 0.8], ...
  'FontSize',15, ...
  'ListboxTop',0, ...
  'String','List of Figures', ...
  'Style','text');
  
figlist.uiBtn = uicontrol('Parent',figlist.fig, ...
  'Units','pixels', ...
  'BackgroundColor',[0.8, 0.8, 0.8], ...
  'FontAngle','italic', ...
  'FontSize',12, ...
  'String','Minimize All', ...
  'Callback','global figlist; grSetVisible(2:(1+size(get(figlist.list, ''String''),1)),0);');
  
figlist.nChars        = 50;
figlist.currentFreeze = 0;

grSetPrinting(0);
grResize;

set(1,'HandleVisibility','off');

figlist.deleted = 0;
set(1,'DeleteFcn','global figlist; figlist.deleted=1;');

% ---------------------------------------------

set(0,'DefaultFigureCreateFcn','grOnCreate');


