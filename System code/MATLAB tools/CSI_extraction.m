function [Rx1,Rx2,sequence,csi] = CSI_extraction (oridata,foldername,format,nRx)
%this function is used to get CSI Mag or Phase data from PicoScenes
%document.Compared to CSIdata this function can output the CSI data on each
%Rx antenna respectively.
%
%oridata:PicoScenes sruct
%format:a string,including:'nonht','vht','ht','hesu','hemu'
%output: Rx1 and Rx2 are 1*2 cell--{amplitude,phase}
%% collect data from thousands of struct consumes a lot of time.
%%In order to save time, the function will save the new output evertime in
%%the folder.
%safe
if nargin ~= 4
    error('Input variable number incorrect!')
end
% foldername=inputname(1);
% foldername = oridata;
filename=format;
%find whether have the folder
n=exist(foldername);
switch n
    case 0  %not have
        mkdir(foldername);%build and enter the folder
        cd(foldername);
    case 7  %already have
        cd(foldername);
        if exist([filename,'-','Rx1','.mat'])==2 && exist([filename,'-','Rx2','.mat'])==2 %whether have saved the data
            Rx1=load([filename,'-','Rx1','.mat']).Rx1;
            Rx2=load([filename,'-','Rx2','.mat']).Rx2;
            sequence=load([filename,'-','sequence','.mat']).sequence;
            csi = load([filename,'-','csi','.mat']).csi;
            %return the existing data
            cd('..')
            return
        end

end
format=lower(format);%transfer into lowercase character
amplitude=[];%initialize output
phase=[];
sequence=[];
csi = [];
%find the format and get the parameter required data
if length(oridata) > 1
    switch format
        case 'nonht'
            for i=1:length(oridata)
                data=oridata{i};
                if isfield(data.PicoScenesHeader,'Version') == 0
                    continue
                end
                pocketformat=getfield(getfield(data,'RxSBasic'),'PacketFormat');
                switch pocketformat
                    case 0
                        CSI_amplitude=getfield(getfield(data,'CSI'),'Mag');
                        CSI_phase=getfield(getfield(data,'CSI'),'Phase');
                        pocketnum = getfield(getfield(data,'StandardHeader'),'Sequence');
                        % ToF
                        Ttime = getfield(getfield(data,'RxSBasic'),'Timestamp');
                        Rtime = getfield(getfield(data,'RxSBasic'),'SystemTime');
                        Rtime = num2str(Rtime);
                        Rtime = Rtime(7:end);
                        Rtime = round(str2num(Rtime)/1000);
                        Ttime = Ttime + 9000000000;
                        interval = Ttime - Rtime;

                        if nRx == 2
                            CSI_amplitude = cat(1,CSI_amplitude(:,:,1),CSI_amplitude(:,:,2));
                            CSI_phase = cat(1,CSI_phase(:,:,1),CSI_phase(:,:,2));
                        end

                        amplitude=cat(1,amplitude,CSI_amplitude);
                        phase=cat(1,phase,CSI_phase);
                        sequence=cat(1,sequence,pocketnum);
                        ToF = cat(1,ToF,interval);
                end
            end
        case 'ht'
            for i=1:length(oridata)
                data=oridata{i};
                if isfield(data.PicoScenesHeader,'Version') == 0
                    continue
                end
                pocketformat=getfield(getfield(data,'RxSBasic'),'PacketFormat');
                switch pocketformat
                    case 1
                        CSI_amplitude=getfield(getfield(data,'CSI'),'Mag');
                        CSI_phase=getfield(getfield(data,'CSI'),'Phase');
                        pocketnum = getfield(getfield(data,'StandardHeader'),'Sequence');
                        CSI = getfield(getfield(data,'CSI'),'CSI');

                        if nRx == 2
                            CSI_amplitude = cat(1,CSI_amplitude(:,:,1),CSI_amplitude(:,:,2));
                            CSI_phase = cat(1,CSI_phase(:,:,1),CSI_phase(:,:,2));
                        end

                        amplitude=cat(1,amplitude,CSI_amplitude');
                        phase=cat(1,phase,CSI_phase');
                        sequence=cat(1,sequence,pocketnum);
                        csi = cat(2,csi,CSI);
                end
            end
        case 'vht'
            for i=1:length(oridata)
                disp(['extractdata',num2str(i/length(oridata)*100),'%']);
                data=oridata{i};
                if isfield(data.PicoScenesHeader,'Version') == 0
                    continue
                end
                pocketformat=getfield(getfield(data,'RxSBasic'),'PacketFormat');
                switch pocketformat
                    case 2
                        CSI_amplitude=getfield(getfield(data,'CSI'),'Mag');
                        CSI_phase=getfield(getfield(data,'CSI'),'Phase');
                        pocketnum = getfield(getfield(data,'StandardHeader'),'Sequence');

                        if nRx == 2
                            CSI_amplitude = cat(1,CSI_amplitude(:,:,1),CSI_amplitude(:,:,2));
                            CSI_phase = cat(1,CSI_phase(:,:,1),CSI_phase(:,:,2));
                        end

                        amplitude=cat(1,amplitude,CSI_amplitude');
                        phase=cat(1,phase,CSI_phase');
                        sequence=cat(1,sequence,pocketnum);
                end
            end
        case 'hesu'
            for i=1:length(oridata)
                disp(['extractdata',num2str(i/length(oridata)*100),'%']);
                data=oridata{i};
                if isfield(data.PicoScenesHeader,'Version') == 0
                    continue
                end
                pocketformat=getfield(getfield(data,'RxSBasic'),'PacketFormat');
                switch pocketformat
                    case 3
                        CSI_amplitude=getfield(getfield(data,'CSI'),'Mag');
                        CSI_phase=getfield(getfield(data,'CSI'),'Phase');
                        pocketnum = getfield(getfield(data,'StandardHeader'),'Sequence');

                        if nRx == 2
                            CSI_amplitude = cat(1,CSI_amplitude(:,:,1),CSI_amplitude(:,:,2));
                            CSI_phase = cat(1,CSI_phase(:,:,1),CSI_phase(:,:,2));
                        end

                        amplitude=cat(1,amplitude,CSI_amplitude');
                        phase=cat(1,phase,CSI_phase');
                        sequence=cat(1,sequence,pocketnum);
                end
            end
        case 'hemu'
            for i=1:length(oridata)
                data=oridata{i};
                if isfield(data.PicoScenesHeader,'Version') == 0
                    continue
                end
                pocketformat=getfield(getfield(data,'RxSBasic'),'PacketFormat');
                switch pocketformat
                    case 4
                        CSI_amplitude=getfield(getfield(data,'CSI'),'Mag');
                        CSI_phase=getfield(getfield(data,'CSI'),'Phase');
                        pocketnum = getfield(getfield(data,'StandardHeader'),'Sequence');

                        if nRx == 2
                            CSI_amplitude = cat(1,CSI_amplitude(:,:,1),CSI_amplitude(:,:,2));
                            CSI_phase = cat(1,CSI_phase(:,:,1),CSI_phase(:,:,2));
                        end

                        amplitude=cat(1,amplitude,CSI_amplitude');
                        phase=cat(1,phase,CSI_phase');
                        sequence=cat(1,sequence,pocketnum);
                end
            end
    end
    csi = [csi(:,:,1);csi(:,:,2)]';
    % if original data only has 1 cell
elseif length(oridata) == 1
    data = oridata{1};
    csi = getfield(getfield(data,'CSI'),'CSI');
    if length(csi) > 1
        amplitude=getfield(getfield(data,'CSI'),'Mag');
        phase=getfield(getfield(data,'CSI'),'Phase');
        sequence = getfield(getfield(data,'StandardHeader'),'Sequence');
    end
end
%safe
if isempty(amplitude)
    cd('..');
    error("Can't find package!")
end
%save
output_amp = amplitude;
output_pha = phase;
if nRx == 2
    l=size(output_amp,2);
    Rx1={output_amp(:,1:0.5*l),output_pha(:,1:0.5*l)};
    Rx2={output_amp(:,1+0.5*l:end),output_pha(:,1+0.5*l:end)};
    save([filename,'-','Rx1'],'Rx1');
    save([filename,'-','Rx2'],'Rx2');%save the data
    save([filename,'-','sequence'],'sequence');
    save([filename,'-','csi'],'csi');
else
    Rx1={output_amp,output_pha};
    Rx2={output_amp,output_pha};
    save([filename,'-','Rx1'],'Rx1');
    save([filename,'-','Rx2'],'Rx2');%save the data
    save([filename,'-','sequence'],'sequence');
    save([filename,'-','csi'],'csi');
end
cd('..');
end
