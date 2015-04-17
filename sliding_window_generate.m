load digitStruct.mat

% sliding window
slide_window = 16;
window_length =32;

image_array = [];
is_digit = [];
for a = 1:size(digitStruct, 2)
    a
    im = imread([digitStruct(a).name]);

    % transforming to grayscale
    im = rgb2gray(im);

    dimension = size(im);
    height = dimension(1);
    width = dimension(2);

    distance = [];

    for digit_n = 1:size(digitStruct(a).bbox, 2)
        distance(digit_n, :, :) = [max(digitStruct(a).bbox(digit_n).top+1, 1), max(digitStruct(a).bbox(digit_n).left+1, 1)];
    end

    count = 1;

    distance_local = [];
    for i = 1:slide_window:height
        for j = 1:slide_window:width

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Make sure that the window box is less than the limits of that
            % image
            if i + window_length > height
                continue
            end

            if j + window_length > width
                continue
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sub_image = im(i:i+window_length-1, j:j+window_length-1);
            % save sub-image to giant array
            image_array(end+1, :, :) = sub_image;

            for digit_n = 1:size(digitStruct(a).bbox, 2)
                distance_local(digit_n, count) = norm(distance(digit_n, :) - [i, j]);
            end
            if count == 246
                max_a = i;
                max_b = j;
            end
            count = count + 1;
        end
    end
    if count > 1
        isdigit_local = zeros(1, count - 1);
        for digit_n = 1:size(digitStruct(a).bbox, 2)
            index = find(distance_local(digit_n, :) == min(distance_local(digit_n, :)));
            isdigit_local(index) = 1;
        end

        is_digit = [is_digit isdigit_local];
    end
end

save('sliding_window.mat', 'image_array', 'is_digit')
quit
