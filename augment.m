% **********************************************************
%                                                          *
% This script augments the street view images data set     *
% by blurring random images and adding them to the set.    *
%                                                          *
% Authors: Dustin Kut Moy Cheung & Didier Landry           *
%                                                          *
% **********************************************************

% Indicate by how many examples to augment the data set.
aug = 25000;

% Select random indices. Randperm ensures indices won't repeat.
rdm     = randperm(size(X,4));
rdm_idx = rdm(1:aug);

X_aug = X;
y_aug = y;

% Gaussian filter with radius 15.
G = fspecial('gaussian',[15 15],2);

% Apply filter and add to array.
for i=1:aug
    % Generate a random angle in the range -5, 5
    angle = randperm(10,1) - 5;
    % Apply gaussian filter and rotate image
    X_aug(:,:,:,size(X_aug,4)+1) = imfilter(...
        imrotate(X(:,:,:,rdm_idx(i)),angle,'crop'),G,'same');
    y_aug(size(y_aug)+1)       = y(rdm_idx(i));
end

% Shuffle array.
rdm_idx = randperm(size(X_aug,4));
X_aug_rdm = X_aug(:,:,:,rdm_idx);
y_aug_rdm = y_aug(rdm_idx);

% Add extras.
X_aug_extr = cat(4, X_extr, X_aug_rdm);
y_aug_extr = cat(1, y_extr, y_aug_rdm);

save('train_32x32_augmented.mat', 'X_aug_rdm', 'y_aug_rdm');
save('train_32x32_augmented_extras.mat', 'X_aug_extr', 'y_aug_extr');