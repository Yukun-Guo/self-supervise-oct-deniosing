# data processing

The Labels_temp is created use the Labels_temp_old by the following matlab code

```bash
files =dir('\data\Labels_temp_old');
dist_folder = '\data\Labels_temp';
for i = 3:length(files)
    img = fullfile(files(i).folder,files(i).name);
    img = imread(img);
    img = imbilatfilt(img);    
    img = imsharpen(img,'Radius',1.2,'Amount',1.1);
    imwrite(img,fullfile(dist_folder,files(i).name));
end
```