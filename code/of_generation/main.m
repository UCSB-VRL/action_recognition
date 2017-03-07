clc; clear variables; close all;

dataset_name = 'UCF-101';

switch dataset_name
    case 'UCF-101'
        dataset_data.root = '/media/brain/archith/video_analysis/UCF-101/UCF-101-images/';
        dataset_data.num_classes = 101;
        dataset_data.image_f_str = 'UCF-101-images';
        dataset_data.optfl_f_str = 'UCF-101-optical_flow';
        
    otherwise
        fprintf('Dataset %s not implemented!!', dataset_name);
        exit;
      
end


class_dirs = dir(dataset_data.root);
class_dirs(1:2)= [];

for class_idx = 1:length(class_dirs)
    class_path = fullfile(class_dirs(class_idx).folder, class_dirs(class_idx).name);
    class_videos = dir(class_path);
    class_videos(1:2) = [];
    
    disp(class_path);
    parfor vid_idx = 1:length(class_videos)
        disp(class_videos(vid_idx).name);
        vid_path = fullfile(class_videos(vid_idx).folder, class_videos(vid_idx).name);
        of_file_path = strcat(strrep(vid_path, dataset_data.image_f_str, dataset_data.optfl_f_str), '.mat');
        if exist(of_file_path, 'file') ~= 0
            continue
        end
        process_video(vid_path, of_file_path);
        
    end
    
    
end

exit;
