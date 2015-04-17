%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Desctiption:
% This script finds digits in an image and returns a box enclosing them
% (all).
%
% Authors: Dustin Kut Moy Cheung, Didier Landry
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [final_box, mask] = getBox(im1, type)
    BW = edge(im1,'sobel', 'horizontal'); %finding horizontal edges 
    BW2 = edge(im1,'sobel','vertical'); %finding vertical edges 
    
    figure,imshow(im1);
    figure,imshow(BW),imshow(BW2);
    %figure,imshow(B);
    %figure,imshow(n1,[]);

    se1=strel('disk',10);
    se2=strel('rectangle',[10,30]);
    % Morphological closing on areas with high levels of horizontal gradients.
    horizMask = imclose(BW,se2);
    % Morphological closing on areas with high levels of vertical gradients.
    vertMask = imclose(BW2,se1);
    % Combine to find areas where there are a lot of vertical AND horizontal
    % gradients.
    if (strcmp(type, 'combined'))
        mask = horizMask .* vertMask;
    elseif (strcmp(type, 'vertical'))
        mask = vertMask; 
    end

    % Filter unmasked areas by excentricity and solidity (number of unmasked
    % pixels in a rectangle surrounding the masked area)
    connComp = bwconncomp(mask); % Find connected components
    stats = regionprops(connComp,'Area','Eccentricity','Solidity');

    mask(vertcat(connComp.PixelIdxList{[stats.Eccentricity] > .95})) = 0;
    mask(vertcat(connComp.PixelIdxList{[stats.Solidity] < .4})) = 0;

    %[i,j]=find(B);
    [is,js] = find(mask);
    [idx,C] = kmeans([is,js],1);

    % Find bounding boxes of large regions.
    areaThreshold = 500; % threshold in pixels
    connComp = bwconncomp(mask);
    stats = regionprops(connComp,'BoundingBox','Area');
    boxes = round(vertcat(stats(vertcat(stats.Area) > areaThreshold).BoundingBox));

    center_of_boxes = zeros(size(boxes,1),2);
    for i=1:size(boxes,1)
        center_of_boxes(i,2) = boxes(i,1)+boxes(i,3)/2;
        center_of_boxes(i,1) = boxes(i,2)+boxes(i,4)/2;
    end

    if (length(C)>1 && length(center_of_boxes)>1)

    % If multiple text region candidates, take the one closest to center of
    % gravity of unmasked area.
    i= dsearchn(center_of_boxes,C);
    
    % See if other boxes have similar dimensions. digits might already be
    % separated.
    for j=1:size(boxes,1)
        if (j~=i)
           if abs(boxes(i,3)-boxes(j,3))<50 && abs(boxes(i,4)-boxes(j,4))<50
               msgbox('Multiple candidates')
           end
        end
    end

    % Dilate box by "factor".
    factor = 0.125;
    final_box = boxes(i,:);
    %boxes: [xmin ymin width height]
    %final_box(1) = round(boxes(i,1)-factor*boxes(i,4));
    %final_box(2) = round(boxes(i,2)-factor*boxes(i,4));
    %final_box(3) = round(boxes(i,3)+2*factor*boxes(i,4));
    %final_box(4) = round(boxes(i,4)+2*factor*boxes(i,4));
    else
        final_box = 0;
    end
end