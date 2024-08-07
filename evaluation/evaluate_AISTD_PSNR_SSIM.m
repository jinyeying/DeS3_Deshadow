%% compute RMSE(MAE)
clear;close all;clc

%maskdir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\AISTD\mask\';
maskdir = 'D:\Dropbox\0pytorch_project_6\dataset\ISTD_ALLINONE\ISTD\test\test_M\';
MD = dir([maskdir '\*.png']);

% result directory
shadowdir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\AISTD\output\'
SD = dir([shadowdir '\*.png']);

% ground truth directory
freedir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\AISTD\gt\'; %AISTD
FD = dir([freedir '\*.png']);

total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(SD,1)); 
smae=zeros(1,size(SD,1)); 
nmae=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    sname = strcat(shadowdir,SD(i).name);
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    
    f = double(f)/255;
    s = double(s)/255;
    
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);
    
    nmask=~m;       %mask of non-shadow region|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    
    ppsnr(i)=psnr(s,f);
    psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% MAE, per pixel
    allmae(i)=sum(dist(:))/size(f,1)/size(f,2);
    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  

    disp(i);
end
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allmae),total_distn/total_pixeln,total_dists/total_pixels);
%input shadow
% PSNR(all,non-shadow,shadow):
% 20.456168	37.456836	20.833840
% SSIM(all,non-shadow,shadow):
% 0.894031	0.984583	0.926625
% PI-Lab(all,non-shadow,shadow):
% 8.396283	2.416566	36.951050
% PP-Lab(all,non-shadow,shadow):
% 8.396283	2.401622	39.012122

% PSNR(all,non-shadow,shadow):
% 31.383625	34.701388	36.491816
% SSIM(all,non-shadow,shadow):
% 0.957507	0.972317	0.989459
% PI-Lab(all,non-shadow,shadow):
% 3.944160	3.390984	6.617972
% PP-Lab(all,non-shadow,shadow):
% 3.944160	3.410026	6.672077