clc;
clear all;
close all;
a=imread('GreyImg.png');
figure(5);
imshow(a);
a=rgb2gray(a);
figure(1);
imshow(a);
[row,col]=size(a); 
d=1500; 
alpha=20;
h=tand(alpha)*d;
[s1,s2]=size(a);
for y=1:s2
    y1(y)=(y/sind(alpha))./(((1-((y/h)*cosd(alpha)))));
    y1=round(y1);
        for t=1:s1
            x1(t,y)=(t*sqrt(((h.^2)+((d+y1(y)).^2))))/sqrt(((h.^2)+(d.^2)+(y.^2)));
        end
end
x1=round(x1);
df=zeros(round(max(x1(:))),round(max(y1(:))));
sub_con=1;
df_indices = cell(s1, s2);

for r1=1:s2
       diff=(x1(s1,s2)-(x1(end,sub_con)))/2;
                    for c1=1:s1
                        re=round(x1(c1,sub_con)+diff);
                        c2=round(y1(1,sub_con));
                        df(re,c2)=a(c1,sub_con);
                        df_indices{c1, r1} = [re, c2];
                        x_new(c1,r1)=re;
                    end
        sub_con=sub_con+1;
end


for y_val=1:size(y1,2)
          for hk=size(x_new,1):-1:2
              df_v=round(x_new(hk,y_val))-round(x_new(hk-1,y_val));
              in_v=round(x_new(hk,y_val));
              ls_v=round(x_new(hk-1,y_val));
               while df_v>1
                  delt_x=in_v-ls_v;
                  delt_y=df(ls_v,round(y1(1,y_val)))-df(in_v,round(y1(1,y_val)));
                  m=round(delt_y./delt_x);
                  df(in_v-1,round(y1(1,y_val)))=round(m+df(in_v,round(y1(1,y_val))));
                  df_v=df_v-1;
                  in_v=in_v-1;
              end
          end
end

if round(y1(1,1))~=1
    for h1=round(y1(1,1))-1:-1:1
    df(:,h1)=df(:,round(y1(1,1)));
    end
end

figure(6);
imshow(uint8(df));

 for y_val=1:size(y1,2)-1
          initial_y=round(y1(1,y_val));
          next_y=round(y1(1,y_val+1));
          dis=round(next_y-initial_y);
          x_var_s=round(x_new(1,y_val));
          x_var_l=round(x_new(end,y_val));
          while dis>1
          for hj=x_var_s:x_var_l
                  delt_x1=next_y-initial_y;
                  delt_y1=df(hj,next_y)-df(hj,initial_y);
                  m1=round(delt_y1./delt_x1);
                  df(hj,initial_y+1)=round(m1+df(hj,initial_y));      
          end
              dis=dis-1;
              initial_y=initial_y+1;
          end
 end
 figure(2);
 imshow(uint8(df));

 tempMat = df;

[row1, col1] = size(df);
k = 1;
for i=1:row1
    for j=1:col1
        if df(i,j) == 0
            black_indices(k,1) = i;
            black_indices(k,2) = j;
            k = k + 1;
        end
    end
end

% Define the secret message
secret_message = 'This is the secret Message which is to be embedded in the original image'; % The message you want to hide

% Convert the message to binary
binary_message = reshape(dec2bin(secret_message, 8).'-'0', 1, []);

% Prepare for embedding: Length of binary message and index for embedding
message_length = length(binary_message);
index = 1;

% Steganography: Embed the message in the anamorphic image (LSB technique)
tuple_array = cell2mat(df_indices(:));
for row = 1:size(df,1)
    for col = 1:size(df,2)
        if index <= message_length
            % Get the pixel value
            pixel_value = df(row, col);
            
            % Modify the least significant bit (LSB) of the pixel with the message bit
            pixInd = [row,col];

            if ~ismember(pixInd, tuple_array, 'rows') && ~ismember(pixInd, black_indices, 'rows')
                pixel_value = bitset(pixel_value, 1, binary_message(index));
                df(row,col) = pixel_value;

                % Move to the next bit in the message
                index = index + 1;
            end
        end
    end
end

figure(3);
imshow(uint8(df));
title('Anamorphic Image with Hidden Message');

 
% % undistorted_image
for i=1:size(df,2)
    y3(i)=(i/((1/sind(alpha))+((i/h)*cosd(alpha))));
end
all_y=unique(round(y3));
all_y=all_y(all_y>0);
for vas=1:size(all_y,2)
    vn=abs(all_y(1,vas)-y3);
    [w1,w2]=min(vn);
    replaced(vas,1)=w2;
end
replaced=replaced';
for j=1:size(all_y,2)
for t=1:size(df,1)
   x2(t,j)=(t*sqrt(((h^2)+(d^2)+(all_y(1,j))^2)))/sqrt(((h^2)+((d+replaced(1,j))^2)));
end
end
x3=1:round(x2(end,end));  
x3=x3';
x2_update=round(x2);
for hj=1:size(x2,2)
    for chv=1:size(x3,1)
    vn=abs(x2(:,hj)-x3(chv,1));
    [w1,w2]=min(vn);
    replaced_x(chv,hj)=w2;
    end
end
sub_con=1;
for r11=1:size(replaced_x,2)
       diff1=round((replaced_x(end,end)-replaced_x(end,r11))/2);
                    for c11=1:size(replaced_x,1)
                        n_x=round(replaced_x(c11,r11)+diff1);
                        update_x(c11,r11)=n_x;
                    end
end
for tc=1:size(replaced,2)
    for tr=1:size(update_x,1)
      tranformed_image(tr,tc)=df(update_x(tr,tc),replaced(1,tc));  
    end
end
figure(4);
imshow(uint8(tranformed_image));

% Steganography: Extract the hidden message from the anamorphic image
extracted_binary_message = [];
index = 1;

for row = 1:size(df,1)
    for col = 1:size(df,2)
        if index <= message_length   
            % Modify the least significant bit (LSB) of the pixel with the message bit
            pixInd = [row,col];
            % Extract the least significant bit (LSB) of the pixel
            if ~ismember(pixInd, tuple_array, 'rows') && ~ismember(pixInd, black_indices, 'rows')
                extracted_bit = bitget(df(row, col), 1);
                 % Append the extracted bit to the binary message
                extracted_binary_message = [extracted_binary_message extracted_bit];
            
                % Move to the next bit in the message
                index = index + 1;
            end
        end
    end
end

% Convert the binary message back to characters
extracted_message = char(bin2dec(reshape(char(extracted_binary_message + '0'), 8, []).')).';

% Display the extracted message
disp(['Extracted Message: ', extracted_message]);

max_characters=((size(df,1)*size(df,2))-(size(tuple_array,1)+size(black_indices,1)))/8;
disp(['Maximum Length of the message: ']);
disp(max_characters);

%SSIM Performance Metric
% Compute SSIM
[ssimval, ssimmap] = ssim(df, tempMat);

% Display results
fprintf('The SSIM value is %0.9f.\n', ssimval);

%Mean Square Error
mse_val = mean((df(:) - tempMat(:)).^2);

% Display MSE value
fprintf('The Mean Squared Error (MSE) is %0.9f.\n', mse_val);

% PSNR formula
psnr_val = 10 * log10((1) / mse_val);

% Display PSNR value
fprintf('The Peak Signal-to-Noise Ratio (PSNR) is %0.9f dB.\n', psnr_val);