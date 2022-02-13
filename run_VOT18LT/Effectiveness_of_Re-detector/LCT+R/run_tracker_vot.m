function run_tracker_vot(results_path)

dataset_path = '/home/space/Documents/LTB35';
% results_path = '/home/space/Documents/experiment/lct-tracker-master/results/longterm';

if ~exist(results_path, 'dir')
    mkdir(results_path);
end

name_list = dir('/home/space/Documents/LTB35');

isub = [name_list(:).isdir]; %# returns logical vector
name_list = name_list(isub);
name_list = name_list(3:end);


parfor i=1:numel(name_list)
    
    show_visualization = false;
    save_results = true;
    
    
    name = name_list(i).name;
    
    fprintf('Processing sequence: %s\n', name);
    results_path2 = fullfile(results_path, name);
    if ~my_exist(results_path2)
        mkdir(results_path2);
    end
    
    bboxes_path = fullfile(results_path2, sprintf('%s_001.txt', name));
    scores_path1 = fullfile(results_path2, sprintf('%s_001_confidence.value', name));
    % check for completeness
    if my_exist(bboxes_path)
        fprintf('Sequence already processed, skipping...\n');
        continue;
    end
    
    gt = dlmread(fullfile(dataset_path, name, 'groundtruth.txt'));
    video_len = size(gt,1);
    % read first image and initialize tracker
    img = imread(fullfile(dataset_path, name, 'color', sprintf('%08d.jpg', 1)));
    gt = gt(1, :);

    % allocate memory for results and store first frame
    bboxes = zeros(video_len, 4);
    bboxes(1,:) = gt;
    scores1 = zeros(video_len, 1);
    idx = 2;
    
    % H W cy cx
    [img_files, pos, target_sz, ground_truth, video_path] = load_video_info_vot(dataset_path, name);
    % tracker
    [positions, confidence] = tracker_lct_vot(video_path, img_files, pos, target_sz, show_visualization);

    if save_results
        my_save(bboxes_path, positions);
        my_save(scores_path1, confidence);
    end

end
end

function ex_ = my_exist(exist_path)
    ex_ = exist(exist_path, 'dir');
end  % endfunction


function my_save(save_results_path, results)
    dlmwrite(save_results_path, results, 'precision', '%.6f');
end  % endfunction
