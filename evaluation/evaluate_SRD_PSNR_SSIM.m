%% compute RMSE(MAE)
clear;close all;clc

maskdir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\SRD\mask\';
MD = dir([maskdir '/*.jpg']);

% result directory
shadowdir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\SRD_AAAI24\'; 
SD = dir([shadowdir '/*.jpg']);
    
% ground truth directory
freedir = 'D:\Dropbox\shadow_results\AAAI2024\DeS3_RESULTS\SRD\gt\';
FD = dir([freedir '/*.jpg']);

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
    %disp(SD(i));
    %disp(FD(i));
    %disp(MD(i));
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,SD(i).name); 
    mname = strcat(maskdir,SD(i).name); 
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
    %ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    %ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    ppsnrs(i)=psnr(s.*(smask),f.*(smask));
    ppsnrn(i)=psnr(s.*(nmask),f.*(nmask));
    
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*(smask),f.*(smask));
    sssimn(i)=ssim(s.*(nmask),f.*(nmask));

    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*(smask);
    sumsdist=sum(sdist(:));
    ndist=dist.*(nmask);
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

% PSNR(all,non-shadow,shadow):
% 34.108160	38.123809	37.451032
% SSIM(all,non-shadow,shadow):
% 0.968059	0.988338	0.984409
