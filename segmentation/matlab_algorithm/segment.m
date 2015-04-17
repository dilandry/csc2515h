%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Desctiption:
% This script finds digits in an image and separates them into 32x32 sub-
% images.
%
% Authors: Dustin Kut Moy Cheung, Didier Landry
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [final] = segment(number,k)
    im=imread(k);
    im = imresize(im, 'OutputSize', [200, NaN]);
    im1=rgb2gray(im);
    im1=medfilt2(im1,[3 3]); %Median filtering of the image to remove noise
    
    % Get general area where digits would be.
    [final_box, mask] = getBox(im1, 'vertical');
    
    % If no boxes were found, increase contrast and repeat...
    while (final_box == 0)
        msgbox('Increasing contrast')
        % Apply contrast-limited adaptive histogram equalization (CLAHE)
        im1 = adapthisteq(im1);
        [final_box, mask] = getBox(im1, 'vertical');
    end
    
    % Show image with overlaid mask.
    displayImage = im;
    displayImage(~repmat(mask,1,1,3)) = 0;
    figure; imshow(displayImage);
    
    % Separate digits. Outputs 32x32 images
    final = separateDigits(im, final_box, number);
end