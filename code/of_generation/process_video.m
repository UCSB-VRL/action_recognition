function return_val = process_video(input_folder, output_location)


% input_folder : Should have images extracted from the video in sequential
% order
% output_location : Output location where the mat file will be stored

image_files = dir(input_folder);
image_files(1:2) = [];

num_images = length(image_files);

tmp_img = imread(fullfile(image_files(1).folder, image_files(1).name));

img_h = size(tmp_img,1);
img_w = size(tmp_img,2);

opticFlow = opticalFlowLK('NoiseThreshold',0.009);

opticFlow_data = zeros(img_h, img_w, 3, num_images);
for i = 1:num_images
   
    frameRGB = imread(fullfile(image_files(i).folder, image_files(i).name));
    frameGray = rgb2gray(frameRGB);
    
    flow = estimateFlow(opticFlow,frameGray);
    
    opticFlow_data(:,:,1, i) = flow.Vx;
    opticFlow_data(:,:,2, i) = flow.Vy;
    opticFlow_data(:,:,3, i) = flow.Magnitude;

end

[output_dir, output_file, ext] = fileparts(output_location);
% Check if output directory already present
if (7 ~= exist(output_dir,'dir'))
    mkdir(output_dir);
end

save(output_location, 'opticFlow_data', '-v7.3');

end