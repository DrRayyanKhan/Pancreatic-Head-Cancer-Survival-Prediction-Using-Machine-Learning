% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synthetic Blurred Images Generation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ROC Curve%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%% Notepad text file read%%%%%%%%%%%%%%%%%%%%%%%%%
AUC0=0.7681;
AUC1=0.8674;
AUC2=0.8629;
path = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Codes/12-05-2022');
S = dir(fullfile(path));
% % Assigning the path
fileName0 = fullfile(path,'fpr0.txt');
fileName1 = fullfile(path,'fpr1.txt');
fileName2 = fullfile(path,'fpr2.txt');
fileName00 = fullfile(path,'tpr0.txt');
fileName11 = fullfile(path,'tpr1.txt');
fileName22 = fullfile(path,'tpr2.txt');

% % Importing all the data from the txt file
fpr0 = importdata(fileName0);
fpr1 = importdata(fileName1);
fpr2 = importdata(fileName2);
tpr0 = importdata(fileName00);
tpr1 = importdata(fileName11);
tpr2 = importdata(fileName22);



% X=[0,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1];
% Y=[0,0.2,0.4,0.6,0.86,0.90,0.98,0.99,1];



% AUC1=trapz(X,Y);
plot(fpr0,tpr0,'Color',[0.6350 0.0780 0.1840],'LineStyle','--','LineWidth',1.5);
hold on;


plot(fpr1,tpr1,'Color',[0.4660 0.6740 0.1880],'LineStyle','-','LineWidth',1.5);
% AUC2=trapz(X,Y1);
hold on;

plot(fpr2,tpr2,'Color',[0 0.4470 0.7410],'LineStyle','-.','LineWidth',1.5);
%AUC3=trapz(X,Y2);


set(gcf,'color','w');
%
ax1 = gca; % current axes
ax1.XTick=0:0.2:1;
ax1.YTick=0:0.2:1;
ax1.XMinorTick='on';
ax1.YMinorTick='on';
ax1.YGrid='on';
ax1.XGrid='on';
ax1.YMinorGrid='off';
ax1.XColor='k';
ax1.YColor='k';
ax1.TickDir='in';
ax1.XAxisLocation='bottom';
ax1.YAxisLocation='left';
ax1.TickLength=[0.01 0.01];
ax1.LineWidth=1;
ax1.FontName='Arial';
ax1.FontSize=12;
ax1.XLim = [0 1];
ax1.YLim = [0 1];
xlabel('False positive rate','FontSize',10);
ylabel('True positive rate','FontSize',10);
plot(fpr0(72),tpr0(72),'r.','MarkerSize',15);
plot(fpr1(87),tpr1(87),'r.','MarkerSize',15);
plot(fpr2(89),tpr2(89),'r.','MarkerSize',15);
legend('HCC Multiple, AUC:0.7681','HCC Single, AUC:0.8674','mCRC, AUC:0.8629','Optimal cut point','Location', [0.58 0.40 0.15 0.05],'FontSize',10);
title('ROC Curves OvR','fontweight','bold');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sensitivity, Precision
% C=[180,6,5;3,301,7;6,5,307];
% %% Plot the confusion matrix C
% figure
% confusionchart(C)
% 
% %% Use the function you've just downloaded
% stats = statsOfMeasure(C, 1);
% 
% %% To access the "false_positive" of the 2nd class
% stats.classes(2,2)
%%
%%%%%%%%%%%%%%%%%%%% Calculate Stats using Confusion matrix only%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculate Accuracy
%  path = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Plato_upload/Categorical_Class/Test');
% fileName = fullfile(path,'Testt.txt');
% Data = importdata(fileName);
% Y_pred = Data(:,1);
% Y_org = Data(:,2);
% cm = confusionchart(Y_org,Y_pred);
% Accuracy = sum(Y_org == Y_pred,'all')/numel(Y_pred);
% Processed = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Final_deblur_dataset/Test/28-01/Out_Refine');
% Reference=sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Final_deblur_dataset/Test/28-01/1/0');
% S = dir(fullfile(Reference,'*.jpg'));
% P = dir(fullfile(Processed,'*.jpg'));
% S = natsortfiles({S.name});
% P = natsortfiles({P.name});
% J=[];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Image Synthetic Blur %%%%%%%%%%%%%%%%%%%%%%
% A={'gaussian','motion'};
% for N=1:numel(S)
%     im=imread(fullfile(Reference,S{N}));
%     v = randi([1,2],1,1);
%     v1 = randi([1,4],1,1);
%     if v==2
%         angle = randi([10,50],1,1);
%         B=randi([10,20],1,1);
%     else
%         angle = randi([1,10],1,1);
%         B=3;
%     end
%     f = fspecial(A{v},B,angle);
%     blur = imfilter(double(im),f,'replicate');
%     imwrite(uint8(blur),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Temperory/Final_deblur_dataset/Kodak/0/%d.jpg',N));
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Image Resizing%%%%%%%%%%%%%%%%%%%%%%
% Reference=sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Final_deblur_dataset/Test/28-01/0/0');
% Reference=sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Literature/Deep Studies/GOPRO_test/sharp');
% S = dir(fullfile(Reference,'*.png'));
% S = natsortfiles({S.name});
% i=523;
% for N=1:numel(S)
%     im=imread(fullfile(Reference,S{N}));
%     im=imresize(im,[256,256]);
%     i=i+1;
%     imwrite(uint8(im),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Final_deblur_dataset/Test/28-01/1/0/%d.jpg',i));
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%% Test, Val Data selection
% Source1     = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Temperory/Final_deblur_dataset/Train/1/0';
% Dest1 = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Temperory/Final_deblur_dataset/Test/1/0';
% Source2     = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Temperory/Final_deblur_dataset/Train/0/0';
% Dest2 = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Temperory/Final_deblur_dataset/Test/0/0';
% n=523;
% FileList = dir(fullfile(Source1, '*.jpg'));
% Index    = randperm(numel(FileList),n);
% for k = 1:n
%   Source = fullfile(Source1, FileList(Index(k)).name);
%   movefile(Source, Dest1);
%   Source = fullfile(Source2, FileList(Index(k)).name);
%   movefile(Source, Dest2);
% end
%%%%%%%%%%%%%%%%%%%%% Images Evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/1/0';
% Processed = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/23_7/27_7/Out_GAN';
% S = dir(fullfile(Reference,'*.tif'));
% S = natsortfiles({S.name});
% P = dir(fullfile(Processed,'*.tif'));
% P = natsortfiles({P.name});
% J0=[];J1=[];J2=[];
% % [bris_model,nique_model]=model(Ref); %%% Model Training
% for N=1:numel(S)
% ref=imread(fullfile(Reference,S{N}));
% pro=imread(fullfile(Processed,P{N}));
% ref=imresize(ref,[128 128]);
% pro=imresize(pro,[128 128]);
% %%%%%%%%%%%%%%%%%%%% References Indices %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%PSNR,SNR,SSIM,Absolute Mean Sharpness Error, Mean Squared error%%%%%%%%%%%%
% if N<=56
% [PSNR,SNR]=psnr(pro,ref);
% Struc=ssim(pro,ref);
% J0=[J0;SNR PSNR Struc];
% Result0=mean(J0);
% end
% if N>56 && N<=187
% [PSNR,SNR]=psnr(pro,ref);
% Struc=ssim(pro,ref);
% J1=[J1;SNR PSNR Struc];
% Result1=mean(J1);
% end
% if N>187 && N<=300
% [PSNR,SNR]=psnr(pro,ref);
% Struc=ssim(pro,ref);
% J2=[J2;SNR PSNR Struc];
% Result2=mean(J2);
% end
% % MSE=immse(pro,ref);
% % AMSE=SHARPNESS(pro,ref);
% % AMEE=ENTROPY(pro,ref);
% %%%%%%%%%%%%%%%%%%%%%Nonreference Indices%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%NIQUE,PIQE,BRISQUE,SSEQ,Entropy, Gradient%%%%%%%%%%%%%%%%%%%%%%% 
% % brisqueI = brisque(pro,bris_model);
% % niqeI = niqe(pro,nique_model);
% % piqeI = piqe(pro);
% % sseq_score=sseq(pro);
% %  J=[J;SNR0 PSNR0 Struc0];%MSE AMSE AMEE brisqueI niqeI piqeI sseq_score];
% % Result=mean(J);
% end
% % Model Training
% function [bris_model,nique_model]=model(Reference)
% % setDir = fullfile(Reference,'GOPR0*','sharp','*.png');
% imds = imageDatastore(Reference,'FileExtensions',{'.png'});
% opinionScores = 100*rand(1,size(imds.Files,1));
% bris_model = fitbrisque(imds,opinionScores'); % Brisque Model Training
% nique_model = fitniqe(imds); % Nique Model Training
% end
% function [sseq_score]=sseq(X)
% X=rgb2gray(X);
% cd 'D:\UofS\OneDrive - University of Saskatchewan\PhD_Thesis\Codes\SSEQ_index\SSEQ\';
% sseq_score=SSEQ(X);
% % cd 'E:\UofS\OneDrive - University of Saskatchewan\PhD_Thesis\data_base\3D-IRCADb\3Dircadb1\3Dircadb1.1\PATIENT_DICOM\';
% end
% % Absolute Mean Sharpness Error
% function [out]=SHARPNESS(X,Y)
% X=rgb2gray(X);Y=rgb2gray(Y);
% [Xmag,Gdir] = imgradient(X,'sobel');
% [Ymag,Gdir] = imgradient(Y,'sobel');
% out=MAE(Xmag,Ymag);
% end
% % Mean Absolute Error
% function [out]=MAE(X,Y)
% out = mae(X-Y);
% end
% % Absolute mean entropy error
% function [out]=ENTROPY(X,Y)
% X=rgb2gray(X);Y=rgb2gray(Y);
% X=entropy(X);
% Y=entropy(Y);
% out=MAE(X,Y);
% end
% %%%%%%%%%%%%%%%%%%%% Read Images and Rescale%%%%%%%%%%%%%%%%
% Source = sprintf('D:/UofS/OneDrive - University of Saskatchewan/Hospital_Data/Liver Cancer/Data Extraction/OneDrive_2020-12-19/Scans Normal/subject 4 female 70 yo normal postchole mild bile duct prominence portal venous 3 mm helical/originals');
% S = dir(fullfile(Source,'*.dcm'));
% S = natsortfiles({S.name});
% for N=1:numel(S)
% %Source1 = fullfile(Source,S(N).name);
% image=read(fullfile(Source,S{N}));
% imshow(image,[])
% pause(0.2);
% % original2=double(imread('72.tif'));
% % original=original2>0==1;
% % ret=rescale(retImg,0,255);
% % L_deblur=double(ret.*original);%double(imread('deblur_fergus.tif'));
% end
% %%%%%%%%%%%%%%%%%%%% Add noise& blurring%%%%%%%%%%%%%%%%%%
% Org=imread(sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/56.tif'));
% I = imnoise(Org,'gaussian',0.1);
% I = imnoise(I,'salt & pepper',0.1);
% H = fspecial('motion',20,45);
% Blur = imfilter(Org,H,'replicate');
% imwrite(uint8(I),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/noisy.tif'));
% imwrite(uint8(Blur),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/blur.tif'));
% Mask = imread(sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/56 (2).tif'));
% imwrite(uint8(double(imresize(I(:,:,1),[512,512])).*(double(Mask))),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/segment.tif'));
% Org=imread(sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/Graphical Abstract.tif'));
%  I=imresize(Org(:,:,1),[512 512]);
% imwrite(uint8(I),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/Graphical Abstract.tif'));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scaled CT image%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scaled_image=imagesc(double(I(:,:,1)));
% colormap('jet');
% colorbar;
% axis off;
% set(gcf,'color','w');
% cd 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Manuscripts_UofS/CAD_review/Medical Image Analysis/Graphical Abstract/';
% saveas(gcf,'Scaled_image.tif');
% C = cov(double(I(:,:,1)));
% Corr_image=imagesc(C);
% colormap('jet');
% colorbar;
% axis off;
% set(gcf,'color','w');
% saveas(gcf,'Corr_image.tif');
% %%%%%%%%%%%% Notepad text file read%%%%%%%%%%%%%%%%%%%%%%%%%
% path = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Codes/12-05-2022');
% S = dir(fullfile(path));
% % Assigning the path
% fileName = fullfile(path,'CAD_cat0.out');
% % Importing all the data from the txt file
% fileEntireDataSet = importdata(fileName);
% % Assigning the first column only
% stats = fileEntireDataSet.data();
% epochs =stats(:,1);
% train_acc = stats(:,2);
% train_loss = stats(:,3);
% val_acc = stats(:,4);
% val_loss = stats(:,5);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Image with Zoomed windows %%%%%%%%%%%%
% Blur = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/GoPro/blur');
% Processed = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/GoPro/out');
% Reference=sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/GoPro/ref');
% Output = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/Images_with_subwindows');
% B = imread(fullfile(Blur,'6.png'));
% P = dir(fullfile(Processed,'*6.*'));
% R = imread(fullfile(Reference,'6.png'));
% B = imresize(B,[256 256]);
% R = imresize(R,[256 256]);
% for N=1:numel(P)
% Source = fullfile(Processed,P(N).name);
% image=imread(Source);
% im=imresize(image,[256 200]);
% 
% figure();
% subimage([0 256],[0 200],uint8(im)),hold on;
% x_r=[12,95];x_y=[200,254];x_w=[130,215];%RWY
% y_r=[110,140];y_y=[22,70];y_w=[85,130];
% 
% subimage([1 84],[201 256],uint8(im(140:180,10:80,:))), hold on;    %R
% subimage([85 169],[201 256],uint8(im(30:90,150:200,:))),hold on;   %W
% subimage([170 255],[201 256],uint8(im(110:170,100:170,:)));        %Y
% 
% rectangle('Position',[x_r(1) y_r(1) x_r(2)-x_r(1) y_r(2)-y_r(1)], 'LineWidth',1, 'EdgeColor','r');
% rectangle('Position',[x_y(1) y_y(1) x_y(2)-x_y(1) y_y(2)-y_y(1)], 'LineWidth',1, 'EdgeColor','w');
% rectangle('Position',[x_w(1) y_w(1) x_w(2)-x_w(1) y_w(2)-y_w(1)], 'LineWidth',1, 'EdgeColor','y');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rectangle('Position',[0 201 84 55], 'LineWidth',2, 'EdgeColor','r');
% rectangle('Position',[85 201 84 55], 'LineWidth',2, 'EdgeColor','w');
% rectangle('Position',[171 201 84 56], 'LineWidth',2, 'EdgeColor','y');
% axis off;
% set(gcf,'color','w');
% myimage = getframe(gca);
% Name = fullfile(Output,P(N).name);%'23_ref.jpg')%
% 
% imwrite(myimage.cdata,Name,'tif');
% close all
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%Image refinement%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blur = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/Lai/blur');
% Processed = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/Lai/out');
% Reference=sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/Lai/ref');
% Output = sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring/August_GAN/Manuscript Deblurring/Manuscript writing/Images Comparison/Images_with_subwindows');
% B = imread(fullfile(Blur,'saturated_01_gyro_02.png'));
% P1 = imread(fullfile(Processed,'saturated_01_gyro_02_GAN_rayyan.jpg'));
% P2 = imread(fullfile(Processed,'saturated_01_gyro_02_refine_rayyan.jpg'));
% R = imread(fullfile(Reference,'saturated_01_gyro_02.png'));
% B = imresize(B,[256 256]);
% R = imresize(R,[256 256]);
% 
% Residual=P1(:,:,1)-P2(:,:,1);
% Residual(Residual<10)=255;
% % I4=uint8(I4>1==1);
% imshow((Residual));
% imwrite(uint8(Residual),fullfile(Output,'saturated_01_gyro_02_residual.png'));
%%%%%%%%%%%%%%%%%%%%% Reading & Writing from two different folders with
%%%%%%%%%%%%%%%%%%%% intensity adjustment
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Third Data Collection';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images';
% cd(Dest);
% S = dir(fullfile(Source,'*'));
% S = natsortfiles({S.name});
% for N=4:numel(S)
%    Sub = dir(fullfile(Source,S{N},'*.dcm'));%
%    mkdir(fullfile(Dest,S{N},'original'));
%    for M=1:numel(Sub)
%      image=dicomread(fullfile(Source,S{N},Sub(M).name));
%      figure();
%      min(image(:))
%      max(image(:))
% %       imshow(image,[40 255]);
%       imshow(image,[1000 1200]);
% %      if min(image(:))<0 && min(image(:))>-1000
% %          imshow(image,[(min(image(:))-(min(image(:))*1.9)) (max(image(:))-(max(image(:))*0.37))]);
% %      elseif min(image(:))<0 && min(image(:))<-1000
% %           imshow(image,[0 (max(image(:))-(max(image(:))*0.80))]);
% %      elseif min(image(:))==0
% %          imshow(image,[(min(image(:))+ 950) (max(image(:))-(max(image(:))*0.37))]);
% %      end
%      myimage = getframe(gca);
%      imwrite(myimage.cdata,fullfile(Dest,S{N},'original',sprintf('%d.tif',M)));
% 
%      close;
% 
%    end
%     
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Delete files
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/rescaled';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/original/images';
% cd(Dest);
% S = dir(fullfile(Source,'*'));
% S = natsortfiles({S.name});
% for N=1:numel(S)
% if isfile(fullfile(Dest,sprintf('%d.tif',N)))==0
%     delete(fullfile(Source,sprintf('%d.tif',N)));
% end
% end
%%%%%%%%%%%%%sERIES OF SCANS Collage%%%%%%%%%%%%%%%
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/mCRC subject 18 66 yo female portal venous phase helical 5  mm/original';
% S = dir(fullfile(Source,'*.tif'));
% S = natsortfiles({S.name});
% image0=imread(fullfile(Source,S{8}));
% image1=imread(fullfile(Source,S{10}));
% image2=imread(fullfile(Source,S{12}));
% image3=imread(fullfile(Source,S{14}));
% image4=imread(fullfile(Source,S{16}));
% montage({image0,image1,image2,image3,image4},'Size', [1 5]);
%%%%%%%%%%%%%Image resize
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Subject 3- Open Dataset';
% S = dir(fullfile(Source,'*.tif'));
% S = natsortfiles({S.name});
% for N=1:numel(S)
%    im=imread(fullfile(Source,S{N}));
%    im=imresize(im,[512 512]);
% % %    RGB = cat(3, im, im, im);
%    imwrite(uint8(im),fullfile(Source,S{N}),'tif');
% end
%%%%%%%%%%%%%%%%%%%%%% Relevant mask search
% Reference = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Subject 3- Open Dataset';
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Final(Rescaled,Selected)/Final Combined Data/Liver/livermasks';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Subject 3- Open Dataset/Seg/org_seg';
% R = dir(fullfile(Reference,'*.tif'));
% R = natsortfiles({R.name});
% for N=1:numel(R)
%     if isfile(fullfile(Source,R{N}))
% %         im=imread(fullfile(Source,R{N}));
% %         im=imresize(im,[512 512]);
% %         %im = imbinarize(im);
% %         imwrite(im,fullfile(Dest,R{N}),'tif');
%     else
%         disp(R{N});
%         delete(fullfile(Reference,R{N}));
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Combine Folders
% Folder1 = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub';
% S=dir(fullfile(Folder1,'*'));
% S=natsortfiles({S.name});
% i=0;
% for X = [3,4,6:26,30:38]%1:numel(S)
%   Folder2 = fullfile(Folder1,S{X});
%   COPY_From = dir(fullfile(Folder2,'original','*.tif'));
%   R = natsortfiles({COPY_From.name});
%   for Y=1:numel(R)
%   im=imread(fullfile(Folder2,'original',R{Y}));
%   im=imresize(im,[512 512]);
%   str = append(sprintf('%d_',i),S{X},R{Y});
%   fullfilename=fullfile(Dest,str);
%   imwrite(uint8(im),fullfilename);
%   i=i+1;
%   end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CAD masks
%%%%%%%%%%%%%%%%%%%%% Mask rescaling
% Folder = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/Liver_com';
% S=dir(fullfile(Folder,'*.tif'));
% S = natsortfiles({S.name});
% i=0;
% for n = 1:numel(S)
% I= imread(fullfile(Folder,S{n}));
% M1=uint8(double(I)>0.05==1&double(I)<0.05==0);
% imwrite((double(I)>0.05==1&double(I)<0.05==0),sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M1/%d.tif',i));
% % M2 = imbinarize(I,'global');
% % imwrite(M2,sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M2/%d.tif',i));
% % % M3 = imbinarize(I,'adaptive');
% % % imwrite(M3,sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M3/%d.tif',i));
% % M4 = imbinarize(M1,'global');
% % imwrite(M4,sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M4/%d.tif',i));
% % % M5 = imbinarize(M1,'adaptive');
% % imwrite(M5,sprintf('D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M5/%d.tif',i));
% i=i+1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ablation DSC evaluation of masks%%
% result=[];
% source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/Test/livermasks/0';
% obtained = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/nondilate_ResU_liver'; 
% S = (dir(fullfile(source, '*.tif')));
% S = natsortfiles({S.name});
% O = (dir(fullfile(obtained, '*.tif')));
% O = natsortfiles({O.name});
% for N = 1:numel(S)
%     A =logical(imbinarize(imread(fullfile(obtained,O{N})),'global'));%obtained
%     B=logical(imread(fullfile(source,S{N})));%original
%     B = imresize(B,[512 512]);
%     [x,y,z]=size(B);
%     if z>1
%     B=B(:,:,2);%original
%     end
%     %B=B>0==1;
% %     figure(1),imshow(A),hold on;
% %     figure(2),imshow(B);
%    
%     %%%%%%%%%%%%%%%%%%%%%%(Dice Coefficient)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     DSC=dice(A,B);%2*(abs(intersect(A,B))./(abs(A)+abs(B)));
%     result=[result;DSC];
%     Mean_result = mean(result);
%     
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Student's t-test Segementation%%%%%%%
% path0 = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/Liver_crossentropy/New Text Document.txt';
% path1 = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/nondilate_ResU_liver/New Text Document.txt';
% % Importing all the data from the txt file
% fileID0 = fopen(path0,'r');
% formatSpec = '%f';
% x = fscanf(fileID0,formatSpec);
% fileID1 = fopen(path1,'r');
% y = fscanf(fileID1,formatSpec);
% %%%%%%%%%%%%%%%%%%%%% Two Sample t-test
% [h,p] = ttest2(x(2:430),y(2:430));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%Combine images for segmentation paper
% liver_path = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/Test/images/0';
% liver_mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Liver/nondilate_ResU_liver';
% tumor_path = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Tumor/Test/images/0';
% tumor_mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/Tumor/nondilate_ResU_tumor';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Segmentatin/data2/nondilate_RMS-Unet';
% t_p = dir(fullfile(tumor_path,'*.tif'));
% t_p = natsortfiles({t_p.name});
% t_m = dir(fullfile(tumor_mask,'*.tif'));
% t_m = natsortfiles({t_m.name});
% l_p = dir(fullfile(liver_path,'*.tif'));
% l_p = natsortfiles({l_p.name});
% l_m = dir(fullfile(liver_mask,'*.tif'));
% l_m = natsortfiles({l_m.name});
% for N=1:numel(t_p)
%     if isfile(fullfile(liver_path,t_p{N}))
%        l_m_index = find(strcmp(l_p, t_p{N}));
%        liver_im = imread(fullfile(liver_path,t_p{N}));
%        liver_im = imresize(liver_im(:,:,1),[512,512]);
%        liver_mk = imread(fullfile(liver_mask,l_m{l_m_index}));
% %        liver_mk = imbinarize(liver_mk,'global');
%        liver_mk = (double(liver_mk)>0.05==1&double(liver_mk)<0.05==0);
%        liver_mk_foreground_black = Process_mask(liver_mk);
%        tumor_im = imread(fullfile(tumor_path,t_p{N}));
%        tumor_im = imresize(tumor_im(:,:,1),[512,512]);
%        tumor_mk = imread(fullfile(tumor_mask,t_m{N}));
% %        tumor_mk = imbinarize(tumor_mk,'global');
%        tumor_mk = (double(tumor_mk)>0.05==1&double(tumor_mk)<0.05==0);
% 
%        image1 = liver_im.*uint8(liver_mk_foreground_black);
%        image2 = liver_im.*uint8(tumor_mk);
% %         A=imfuse(image1,image2,'falsecolor','Scaling','none','ColorChannels',[1 0 2]);
%        A=imfuse(tumor_mk,liver_mk,'falsecolor','Scaling','none','ColorChannels',[2 1 0]);%[1 0 2]
%        A(:,:,1)=A(:,:,1)+image1;
%        A(:,:,2)=A(:,:,2)+image1;
%        A(:,:,3)=A(:,:,3)+image1;
%        imwrite(A,fullfile(Dest,t_p{N}));
%     end
%     
%     
%     
%     
% end
% function im1 = Process_mask(im)
%     im1 = im2uint8(im);
%     im1 = imresize(im1,[512,512]);
%     im1 = (im1<20==1 & im1>50==0);  
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Manual Segmentation
% path = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub/Test_seg/M1';
% S = dir(fullfile(path,'*.tif'));
% S = natsortfiles({S.name});
% names = dir(fullfile(Dest,'*.tif'));
% names = natsortfiles({names.name});
% for N=[606:615]
% im = imread(fullfile(path,S{N}));
% imshow(uint8(im));
% mask=roipoly;
% mask = double(im2bw(mask));
% %[BW,maskedImage] = segmentImage(im);
% % imageSegmenter(im);
% BW = activecontour(im(:,:,1),mask,500,'edge','SmoothFactor',3,'ContractionBias',-0.4);
% imwrite((BW),fullfile(Dest,names{N}));
% close all;
% 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Test_seg/M1';
% Source = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Combine_sub/Liver_seg';
% S = dir(fullfile(Source,'*.tif'));
% S = natsortfiles({S.name});
% for N=1:numel(S)
%     if isfile(fullfile(Source,S{N}))
%         image=double(imread(fullfile(Source,S{N})));
%         image=imresize(image,[512 512]);
%         s = size(image);
%         a = size(s);
%         mask=double(imread(fullfile(Mask,S{N})));
%         mask=mask(:,:,1);
% %         mask=mask>1==1;
%         image(:,:,1) = image(:,:,1).*mask;
%         if a(2)==3
%         image(:,:,2) = image(:,:,2).*mask;
%         image(:,:,3) = image(:,:,3).*mask;
%         end
%         %im = imbinarize(im);
%         imwrite(uint8(image),fullfile(Dest,S{N}),'tif');
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%Develop Masks
% Image = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub';
% Mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub/Test_seg/M1';
% Dest = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Combine_sub/Liver_seg';
% Images = dir(fullfile(Image,'*.tif'));
% Images = natsortfiles({Images.name});
% Masks = dir(fullfile(Mask,'*.tif'));
% Masks = natsortfiles({Masks.name});
% for N=1:numel(Images)
%     im= double(imread(fullfile(Image,Images{N})));
%     im=imresize(im,[512,512]);
%     s = size(im);
%     a = size(s);
%     mk= double(imread(fullfile(Mask,Masks{N})));
%     mk=imresize(mk,[512,512]);
%     im_seg = im.*mk;
%     im(:,:,1) = im(:,:,1).*mk;
%     if a(2)==3
%         im(:,:,2) = im(:,:,2).*mk;
%         im(:,:,3)=im(:,:,3).*mk;
%     end
%         imwrite(uint8(im_seg),fullfile(Dest,Images{N}));
%     
% end
%%%%%%%%%%%%%%%%%%%%% Make liver mask and separate tumor and non tumor livers
% Malignant_liver = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Final(Rescaled,Selected)/Final Combined Data/Tumor/images';
% li_mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Final(Rescaled,Selected)/Final Combined Data/Liver/livermasks';
% nonMalignant_liver = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Final(Rescaled,Selected)/Final Combined Data/Liver/images';
% Dest_m_li_mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Abnormal_images/Subject 3-Public-Abnormal';
% Dest_nonm_li_mask = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/CAD/Scans Normal_images/Subject 3-Public-Normal';
% 
% nonM_li = dir(fullfile(nonMalignant_liver,'*.tif'));
% nonM_li = natsortfiles({nonM_li.name});
% for N=21357:numel(nonM_li)
%     if isfile(fullfile(Malignant_liver,nonM_li{N}))
%         image=double(imread(fullfile(nonMalignant_liver,nonM_li{N})));
%         image=imresize(image,[512 512]);
%         s = size(image);
%         a = size(s);
%         mask=double(imread(fullfile(li_mask,nonM_li{N})));
%         mask=imresize(mask,[512 512]);
%         mask=mask>1==1;
%         image(:,:,1) = image(:,:,1).*mask(:,:,1);
%         if a(2)==3
%         image(:,:,2) = image(:,:,2).*mask(:,:,1);
%         image(:,:,3) = image(:,:,3).*mask(:,:,1);
% %         image = cat(3, image(:,:,1), image(:,:,2), image(:,:,3));
%         end
%         imwrite(uint8(image),fullfile(Dest_m_li_mask,nonM_li{N}),'tif');
%     else
%         image=double(imread(fullfile(nonMalignant_liver,nonM_li{N})));
%         image=imresize(image,[512 512]);
%         s = size(image);
%         a = size(s);
%         mask=double(imread(fullfile(li_mask,nonM_li{N})));
%         mask=imresize(mask,[512 512]);
%         mask=mask>1==1;
%         image(:,:,1) = image(:,:,1).*mask(:,:,1);
%         if a(2)==3
%         image(:,:,2) = image(:,:,2).*mask(:,:,1);
%         image(:,:,3) = image(:,:,3).*mask(:,:,1);
% %         image = cat(3, image(:,:,1), image(:,:,2), image(:,:,3));
%         end
%         imwrite(uint8(image),fullfile(Dest_nonm_li_mask,nonM_li{N}),'tif');
%     end
% end


%%%%%%%%%%%%%%%%%%%% make dataset for selected images
% CAD_dataset = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/CAD_dataset';
% Raw_images = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/Final Combined/Final Combined/Tumor/images';
% Enhanced_images = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/Final(Rescaled,Selected)/Final Combined Data/Liver/images';
% masks = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/Final(Rescaled,Selected)/Final Combined Data/Liver/livermasks';
% Tumor_images = 'D:/UofS/OneDrive - University of Saskatchewan/PhD_Thesis/Research_Papers/Deblurring_biomedical/Dataset/Final(Rescaled,Selected)/Final Combined Data/Tumor/images';
% mkdir(fullfile(CAD_dataset,'Raw_images_CAD','Raw_tumor'));
% mkdir(fullfile(CAD_dataset,'Raw_images_CAD','Raw_non_tumor'));
% mkdir(fullfile(CAD_dataset,'Enhanced_images_CAD','En_tumor'));
% mkdir(fullfile(CAD_dataset,'Enhanced_images_CAD','En_non_tumor'));
% Raw_set = dir(fullfile(Raw_images,'*.tif'));
% Raw_set = natsortfiles({Raw_set.name});     %All raw images
% En_set = dir(fullfile(Enhanced_images,'*.tif'));
% En_set = natsortfiles({En_set.name});     %All enhanced images
% for N=1:numel(En_set)
% if isfile(fullfile(Tumor_images,En_set{N}))==1
%   im = double(imread(fullfile(Raw_images,En_set{N})));
%   im = imresize(im(:,:,1),[512 512]);
%   msk = double(imread(fullfile(masks,En_set{N})));
%   msk = double(msk(:,:,1));
%   im = (im.*msk);
%   Dest = fullfile(CAD_dataset,'Raw_images_CAD','Raw_tumor',En_set{N});
%   imwrite(uint8(im),Dest);
%   im1 = double(imread(fullfile(Enhanced_images,En_set{N})));
%   im1 = imresize(im1(:,:,1),[512 512]);
%   im1 = (im1.*msk);
%   Dest1 = fullfile(CAD_dataset,'Enhanced_images_CAD','En_tumor',En_set{N});
%   imwrite(uint8(im1),Dest1);
% end
% if isfile(fullfile(Tumor_images,En_set{N}))==0
%   im = double(imread(fullfile(Raw_images,En_set{N})));
%   im = imresize(im(:,:,1),[512 512]);
%   msk = double(imread(fullfile(masks,En_set{N})));
%   msk = double(msk(:,:,1));
%   im = (im.*msk);
%   Dest = fullfile(CAD_dataset,'Raw_images_CAD','Raw_non_tumor',En_set{N});
%   imwrite(uint8(im),Dest);
%   im1 = double(imread(fullfile(Enhanced_images,En_set{N})));
%   im1 = imresize(im1(:,:,1),[512 512]);
%   im1 = (im1.*msk);
%   Dest1 = fullfile(CAD_dataset,'Enhanced_images_CAD','En_non_tumor',En_set{N});
%   imwrite(uint8(im1),Dest1);
% end
% end











