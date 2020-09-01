clear;
clc;

% number of files in the directory
num_of_file = 34;

raw_file = dir('*.mat');

for i = 1:num_of_file
    load(raw_file(i).name);
    Ia_angle = Ia_angle.Data(240:end,:);
    Ia_freq = Ia_freq.Data(240:end,:);
    Ia_mag = Ia_mag.Data(240:end,:);
    
    Ib_angle = Ib_angle.Data(240:end,:);
    Ib_freq = Ib_freq.Data(240:end,:);
    Ib_mag = Ib_mag.Data(240:end,:);
      
    Ic_angle = Ic_angle.Data(240:end,:);
    Ic_freq = Ic_freq.Data(240:end,:);
    Ic_mag = Ic_mag.Data(240:end,:);
       
    Va_angle = Va_angle.Data(240:end,:);
    Va_freq = Va_freq.Data(240:end,:);
    Va_mag = Va_mag.Data(240:end,:);
        
    Vb_angle = Vb_angle.Data(240:end,:);
    Vb_freq = Vb_freq.Data(240:end,:);
    Vb_mag = Vb_mag.Data(240:end,:);
    
    Vc_angle = Vc_angle.Data(240:end,:);
    Vc_freq = Vc_freq.Data(240:end,:);
    Vc_mag = Vc_mag.Data(240:end,:);
    
    Ia_angle_norm = normalize(Ia_angle, 'range',[0,1]);
    Ia_freq_norm = normalize(Ia_freq, 'range',[0,1]);
    Ia_mag_norm = normalize(Ia_mag, 'range',[0,1]);
    
    Ib_angle_norm = normalize(Ib_angle, 'range',[0,1]);
    Ib_freq_norm = normalize(Ib_freq, 'range',[0,1]);
    Ib_mag_norm = normalize(Ib_mag, 'range',[0,1]);
    
    Ic_angle_norm = normalize(Ic_angle, 'range',[0,1]);
    Ic_freq_norm = normalize(Ic_freq, 'range',[0,1]);
    Ic_mag_norm = normalize(Ic_mag, 'range',[0,1]);
    
    Va_angle_norm = normalize(Va_angle, 'range',[0,1]);
    Va_freq_norm = normalize(Va_freq, 'range',[0,1]);
    Va_mag_norm = normalize(Va_mag, 'range',[0,1]);
    
    Vb_angle_norm = normalize(Vb_angle, 'range',[0,1]);
    Vb_freq_norm = normalize(Vb_freq, 'range',[0,1]);
    Vb_mag_norm = normalize(Vb_mag, 'range',[0,1]);
    
    Vc_angle_norm = normalize(Vc_angle, 'range',[0,1]);
    Vc_freq_norm = normalize(Vc_freq, 'range',[0,1]);
    Vc_mag_norm = normalize(Vc_mag, 'range',[0,1]);
    
    norm_matrix = [Ia_angle_norm, Ia_freq_norm, Ia_mag_norm,...
        Ib_angle_norm, Ib_freq_norm, Ib_mag_norm, ...
        Ic_angle_norm, Ic_freq_norm, Ic_mag_norm, ...
        Va_angle_norm, Va_freq_norm, Va_mag_norm, ...
        Vb_angle_norm, Vb_freq_norm, Vb_mag_norm, ...
        Vc_angle_norm, Vc_freq_norm, Vc_mag_norm];
    flag = strfind(raw_file(i).name,'.m') - 1;
    fname = strcat(raw_file(i).name(1:flag),'.csv');
    csvwrite(fname, norm_matrix)
    
end
