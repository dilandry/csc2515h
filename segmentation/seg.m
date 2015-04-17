%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Desctiption:
% This script finds digits in the input images and separates them into 
% 32x32 sub-images.
%                                                           
% Authors: Dustin Kut Moy Cheung, Didier Landry
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set input parameters
% Image number range to process...
from = 3084; to = 3100;
% Folder in which the images are located...
folder = '../data/test/';
final = cell(to-from,1);

%% Loop through the images.
% For a nice one, uncomment this:
%for i=2012:2023
for i=from:to
    [final{i}] = segment(i,strcat(folder,num2str(i),'.png'));
    
    % Close all figures.
    delete(findall(0,'Type','figure')), bdclose('all')  
end

