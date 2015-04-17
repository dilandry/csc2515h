%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Desctiption:
% This functions separates the digits in the input image it is
% given into individual 32x32 images 
%
% Authors: Dustin Kut Moy Cheung, Didier Landry
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [final] = separateDigits(im, final_box, number)
    % Display segmented digit area
    im_crop = imcrop(im, final_box); 
    
    im_crop_gray = rgb2gray(im_crop);
    % Apply Otsu's thresholding method.
    im_crop_bw   = im2bw(im_crop_gray, graythresh(im_crop_gray));

    edges_bw = edge(im_crop_gray,'sobel', 'horizontal');
    
    % If more white than black on edges of image, invert it.
    if sum(sum(sum(im_crop_bw(1:5,:))) + ...
        sum(sum(im_crop_bw(end-4:end,:))) + ...
         sum(sum(im_crop_bw(:,1:5))) +...
          sum(sum(im_crop_bw(:,end-4:end)))) > 5*(size(im_crop_bw,1)+size(im_crop_bw,2))
        im_crop_bw = ~im_crop_bw;
    end
    
    histo = zeros(1, size(im_crop_bw,2));
    for i=1:size(im_crop_bw,2)
        histo(i) = sum(im_crop_bw(:,i));
    end

    % Find minimas (corresponding to spaces between digits)
    maxValue = double(max(im_crop_bw(:)));   % Find the maximum pixel value
    N = 4*max(min(sum(im_crop_bw)),2.5);     % Threshold number of white pixels
    boxIndex2 = (histo) > N*maxValue;

    % Find groups of ones
    D = diff([0,boxIndex2,0]);
    b.beg = find(D == 1);
    b.end = find(D == -1) - 1;
    b.center = b.beg + round((b.end - b.beg)/2);
    
    figure, subplot(3,1,1), imshow(im_crop_bw)
    hold all
    subplot(3,1,2), imshow(edges_bw)
    subplot(3,1,3), plot(1:size(histo,2), histo,'b',...
                         1:size(histo,2),N*maxValue*ones(size(histo)) , 'r');
    
    % Separate the individual digits and save 32x32 images
    figure
    hold all
    final = {};
    for i=1:size(b.beg,2)
        if b.end(i)-b.beg(i) >= 8 ...% Reject boxes that are less than 8 pixels wide
           && i<=4                   % Addresses never have more than 4 digits
            subplot(1,size(b.beg,2),i), imshow(im_crop_bw(:,b.beg(i):b.end(i)));
            height = final_box(4);
            temp = im(final_box(2):final_box(2)+final_box(4)-1,...
                       max(0,final_box(1)+b.center(i)-round(height/2)):...
                       min(size(im,2)-1,final_box(1)+b.center(i)+round(height/2)));
            final{i} = imresize(temp, 'OutputSize', [32, 32]);
            imwrite(final{i}, strcat(num2str(number),'c',num2str(i),'.png'))
        end
    end
    
    % Write original image and unsegmented cropped image
    imwrite(im, strcat(num2str(number),'.png'));
    imwrite(im_crop, strcat(num2str(number),'c','.png'));
end