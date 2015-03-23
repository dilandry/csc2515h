% **********************************************************
%                                                          *
% This script augments the street view images data set     *
% by blurring random images and adding them to the set.    *
%                                                          *
% Authors: Dustin Kut Moy Cheung & Didier Landry           *
%                                                          *
% **********************************************************

% Indicate by how many examples to augment the data set.
aug = 20;

% Select random indices. Randperm ensures indices won't repeat.
rdm     = randperm(size(X,4));
rdm_idx = rdm(1:aug);

X_aug = X;
y_aug = y;

% Gaussian filter with radius 15.
G = fspecial('gaussian',[15 15],2);

for i=1:aug
    X_aug(:,:,:,size(X_aug,4)+1) = imfilter(X(:,:,:,rdm_idx(i)),G,'same');
    y_aug(size(X2,4)+1)       = y(rdm_idx(i));
end

save('train_32x32_augmented.mat', 'X_aug', 'y_aug');