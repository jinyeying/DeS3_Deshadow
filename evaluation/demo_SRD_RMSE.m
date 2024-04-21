clear;
tic
fd = fopen('SRD.txt');
a = textscan(fd, '%s');
fclose(fd);
testfnlist = a{1};

fprintf('Starting evaluation. Total %d images\n', numel(testfnlist));

total_dist_l2 = zeros(1, numel(testfnlist));
total_dist_l4 = zeros(1, numel(testfnlist));
total_dist_l6 = zeros(1, numel(testfnlist));
total_pix_l2 = zeros(1, numel(testfnlist));
total_pix_l4 = zeros(1, numel(testfnlist));
total_pix_l6 = zeros(1, numel(testfnlist));

for recovery_count = 1 : numel(testfnlist)
    % Methods by Guo
    gt_recovery         = imread(['D:\Dropbox\shadow_results\after_ICCV21\SRD_REMOVAL_RESULTS\free256\' testfnlist{recovery_count}(1:end-4) '.png']);
    shadow_recovery     = imread(['D:\Dropbox\shadow_results\after_ICCV21\SRD_REMOVAL_RESULTS\shadow256\' testfnlist{recovery_count}(1:end-4) '.png']); 
    recovered_recovery  = imread(['D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\SRD_AAAI24\' testfnlist{recovery_count}(1:end-4)  '.jpg']);
    m                   = imread(['D:\Dropbox\shadow_results\after_ICCV21\SRD_REMOVAL_RESULTS\mask256\' testfnlist{recovery_count}(1:end-4) '.png']);
    
    gt_recovery         = imresize(gt_recovery,[256 256]);
    shadow_recovery     = imresize(shadow_recovery,[256 256]);
    recovered_recovery  = imresize(recovered_recovery, [256,256]);         
    m=imresize(m,[256 256]);
         
    if numel(size(m)) == 3
        m = rgb2gray(m);
    end
    
    m(m~=0)=1;
    
    m = double(m);
    
    mask_recovery = m;

    mask2_recovery = 1-m;
    
    % for the overall regions
    [total_dist_l2(1, recovery_count), ...
     total_pix_l2(1, recovery_count), ...
     total_dist_l4(1, recovery_count), ...
     total_pix_l4(1, recovery_count), ...
     total_dist_l6(1, recovery_count), ...
     total_pix_l6(1, recovery_count)] = evaluate_recovery(gt_recovery, ...
                                                          recovered_recovery, ...
                                                          NaN*ones(size(gt_recovery)),...
                                                          mask_recovery, ...
                                                          mask2_recovery);
    
end

dist_12 = sum(total_dist_l2(:))/sum(total_pix_l2(:));
dist_14 = sum(total_dist_l4(:))/sum(total_pix_l4(:));
dist_16 = sum(total_dist_l6(:))/sum(total_pix_l6(:));

[dist_14 dist_16 dist_12]
fprintf('%s/%.2f/%s/%.2f/%s/%.2f\n', 'S', dist_14, 'NS', dist_16, 'Overall', dist_12);
%for the shadow evaluation, non_shadow, overall
fprintf('Evaluation complete! Total %d images in %.2f mins\n', numel(testfnlist), toc/60);
%S/5.88/NS/2.83/Overall/3.72