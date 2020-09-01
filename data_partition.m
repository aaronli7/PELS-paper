% The raw dataset partition for diagnosis and detection
clc;
clear;
%% public parameters
% detection: normal(0) abnormal(1)
% diagnosis: normal(0) cyberattack(f1) dc/dc(f2) replay(f3)
% capacitor(f4) delay(f5) PWM(f6)

% partition parameters, decided by the user
% window size 
window_size = 10;

% step size when window moves
s_step = 0.2 * window_size;

% specify the source folder: normal(1834),dcdc_fault(1921), vsc_delay(180)
%% create specific dataset
current_directory = strcat('w',int2str(window_size),'_dataset_new');
mkdir(current_directory)
mkdir(current_directory,'fault_detection')
mkdir(current_directory,'fault_diagnosis')

mkdir(strcat(current_directory, '/fault_detection'), 'Normal')
mkdir(strcat(current_directory, '/fault_detection'), 'Abnormal')

mkdir(strcat(current_directory, '/fault_diagnosis'), 'Normal')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_1')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_2')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_3')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_4')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_5')
mkdir(strcat(current_directory, '/fault_diagnosis'), 'Fault_6')

%% normal partition
filename_detection_normal = strcat('./', current_directory, '/fault_detection/Normal/');
filename_detection_abnormal = strcat('./', current_directory, '/fault_detection/Abnormal/');
filename_diagnosis_normal = strcat('./', current_directory, '/fault_diagnosis/Normal/');

num_of_file = 33;

raw_files = dir('./tocsvdata_new/*.csv');
for i = 1:num_of_file
    ref_matrix = csvread(strcat('./tocsvdata_new/', raw_files(i).name));
    Ia_angle = ref_matrix(:,1);
    Ia_freq = ref_matrix(:,2);
    Ia_mag = ref_matrix(:,3); two
    
    Ib_angle = ref_matrix(:,4);
    Ib_freq = ref_matrix(:,5);
    Ib_mag = ref_matrix(:,6); 
    
    Ic_angle = ref_matrix(:,7);
    Ic_freq = ref_matrix(:,8);
    Ic_mag = ref_matrix(:,9); 
    
    Va_angle = ref_matrix(:,10);
    Va_freq = ref_matrix(:,11);
    Va_mag = ref_matrix(:,12);
    
    Vb_angle = ref_matrix(:,13);
    Vb_freq = ref_matrix(:,14);
    Vb_mag = ref_matrix(:,15);
    
    Vc_angle = ref_matrix(:,16);
    Vc_freq = ref_matrix(:,17);
    Vc_mag = ref_matrix(:,18);
    
    id = str2double(regexp(raw_files(i).name, '\d+', 'match'));
    if isempty(id)
        disp('this is normal')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Normal/');
    
    elseif id==1 || id==2 || id==4 || id==5 || id==6 || id==9 || id==10 ...
            || id==11 || id==12 || id==13 || id==14 || id==15 ...
            || id==16 || id==17 || id==18 || id==19 || id==20 ...
            || id==21 || id==22 || id==23 || id==24 ...
            || id==33 || id==34 || id==36
        disp('it is cyber attack');
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_1/');
        
    elseif id == 29 
        disp('this is delay attack')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_5/');
        
    elseif id == 30
        disp('this is capacitor')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_4/');

    elseif id == 31
        disp('this is PWM')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_6/');
        
    elseif id == 32 || id == 35
        disp('this is dc/dc')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_2/');
        
    elseif id == 27
        disp('this is replay attack')
        disp(raw_files(i).name)
        filename_diagnosis = strcat('./', current_directory, '/fault_diagnosis/Fault_3/');        
    end
    
    for window_start=1:s_step:4563 - window_size + 1

        frame_Ia_angle = Ia_angle(window_start:window_start + window_size - 1);
        frame_Ia_freq = Ia_freq(window_start:window_start + window_size - 1);
        frame_Ia_mag = Ia_mag(window_start:window_start + window_size - 1);

        frame_Ib_angle = Ib_angle(window_start:window_start + window_size - 1);
        frame_Ib_freq = Ib_freq(window_start:window_start + window_size - 1);
        frame_Ib_mag = Ib_mag(window_start:window_start + window_size - 1);

        frame_Ic_angle = Ic_angle(window_start:window_start + window_size - 1);
        frame_Ic_freq = Ic_freq(window_start:window_start + window_size - 1);
        frame_Ic_mag = Ic_mag(window_start:window_start + window_size - 1);

        frame_Va_angle = Va_angle(window_start:window_start + window_size - 1);
        frame_Va_freq = Va_freq(window_start:window_start + window_size - 1);
        frame_Va_mag = Va_mag(window_start:window_start + window_size - 1);

        frame_Vb_angle = Vb_angle(window_start:window_start + window_size - 1);
        frame_Vb_freq = Vb_freq(window_start:window_start + window_size - 1);
        frame_Vb_mag = Vb_mag(window_start:window_start + window_size - 1);

        frame_Vc_angle = Vc_angle(window_start:window_start + window_size - 1);
        frame_Vc_freq = Vc_freq(window_start:window_start + window_size - 1);
        frame_Vc_mag = Vc_mag(window_start:window_start + window_size - 1);
        
        frame_matrix = [frame_Ia_angle, frame_Ia_freq, frame_Ia_mag, ...
            frame_Ib_angle, frame_Ib_freq, frame_Ib_mag, ...
            frame_Ic_angle, frame_Ic_freq, frame_Ic_mag, ...
            frame_Va_angle, frame_Va_freq, frame_Va_mag, ...
            frame_Vb_angle, frame_Vb_freq, frame_Vb_mag, ...
            frame_Vc_angle, frame_Vc_freq, frame_Vc_mag, ...
            ];
        
        if isempty(id)
            fname = strcat(filename_detection_normal, int2str(window_start),'__',raw_files(i).name); 
            csvwrite(fname, frame_matrix)
            
            fname = strcat(filename_diagnosis_normal, int2str(window_start),'__',raw_files(i).name); 
            csvwrite(fname, frame_matrix)
        elseif id == 27
            if window_start + window_size - 1 > 1560 && window_start < 2160
                fname = strcat(filename_detection_abnormal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
           
            else
                fname = strcat(filename_detection_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
            end
           
        elseif id==29 || id == 30 || id==31
            if window_start + window_size - 1 > 1560 
                fname = strcat(filename_detection_abnormal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
            else
                fname = strcat(filename_detection_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
            end
        else
            if window_start + window_size - 1 > 1560 && window_start < 2760
                fname = strcat(filename_detection_abnormal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
            else
                fname = strcat(filename_detection_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
                
                fname = strcat(filename_diagnosis_normal, int2str(window_start),'__',raw_files(i).name);
                csvwrite(fname, frame_matrix);
            end
        end
    end
end