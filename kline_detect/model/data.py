import numpy as np
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset

def normalize_complex(data, eps=0.):
    normalized_data = np.empty_like(data)
    for i in range(data.shape[0]):
        slice_2d = data[i]  # Get the 2D slice at index i
        mag = np.abs(slice_2d)
        mag_std = mag.std()
        normalized_data[i] = slice_2d / (mag_std + eps)
    return normalized_data
def get_all_sample(data_path,patient_list):
    sample_list=[]
    for patient in patient_list:
        patient_path=data_path/patient
        for file in patient_path.iterdir():
            data=np.load(file)
            for i in range(data.shape[0]):
                file_index=[file]+[i]
                sample_list.append(file_index)   
    return sample_list

class Corruptdata(Dataset):
    def __init__(self, dir,mask_dir,window_size, patient_list):
        super(Corruptdata, self).__init__()
        self.patient_list = patient_list
        self.dataroot=Path(dir)
        self.sliceroot = Path(dir)/'Fullsample_slice'
        self.Heartregion_root=Path(dir)/'HeartRegion'
        self.Noise_root=Path(mask_dir)
        self.samplelist = get_all_sample(self.Noise_root,self.patient_list)
        self.sample_num = len(self.samplelist)
        self.window_size=window_size
        
    def transform(self,kdata,center,file_info=''):
        img = np.fft.ifft2(kdata,axes=(-2,-1))
        img = np.fft.fftshift(img,axes=(-2,-1))
        if center[0]-64 < 0 or center[0]+64 > img.shape[1] or center[1]-64 < 0 or center[1]+64 > img.shape[2]:
            raise ValueError(f'{file_info} failed Crop indices are out of bounds')
        else:
            crop_img = img[:,center[0]-64:center[0]+64, center[1]-64:center[1]+64]
        out_k=np.fft.fft2(crop_img,axes=(-2,-1))
        return out_k
    def gen_crop_k_and_target(self,kdata,noise_mask):
        def get_window_start_indices(mask):
            """
            Get the starting indices of each window in the mask.

            Parameters:
                mask (numpy.ndarray): Input mask array.

            Returns:
                list: List containing the starting indices of each window.
            """
            window_start_indices = []
            window_size = self.window_size
            i = 0
            while i < mask.shape[0]:
                # 如果当前行不全为零，则跳过
                if not np.all(mask[i] == 0):
                    i += 1
                    continue
                
                # 找到当前零行块的结束位置
                end = i
                while end < mask.shape[0] and np.all(mask[end] == 0):
                    end += 1
                
                # 如果该零行块大小符合要求，则将起始位置添加到列表中
                if end - i == window_size:
                    window_start_indices.append(i)
                elif end-i > window_size:
                    window_start_indices.append(i)
                    window_start_indices.append(end-self.window_size)
                
                # 将索引移动到下一个零行块的开始位置
                i = end + 1
            
            return window_start_indices

        noise_indice=get_window_start_indices(noise_mask)
        noise_mask=np.stack((noise_mask,noise_mask),axis=0).astype(np.int32)
        noise_mask=np.repeat(noise_mask[:, np.newaxis, :, :], 12, axis=1)

        noise_k=kdata*noise_mask
    
        for i in range(noise_k.shape[1]):
            if i ==0:
                other_indices = [1,2]
            elif i== 11 or 10:
                other_indices = [9,10]
            else:
                other_indices = [i+1,i+2]
            for j in noise_indice:
                random_index = np.random.choice(other_indices)
               
        
                 # 从其他索引中随机选择一个
                
               
                noise_k[:, i, j:j+self.window_size,:] = kdata[:, random_index, j:j+self.window_size,:]
        return noise_k.astype(np.float32)
        
        

    def __getitem__(self, index):
        def get_full_sample_path(mask_path):
            # 转换为 Path 对象
        
            
            # 定义目标路径的基本模板
            base_template = mask_path.parent.parent.parent

            # 从给定路径中提取患者 ID
            patient_id = mask_path.parent.name
            import re
            matches = re.match(r'([a-zA-Z_]+)(_\d+)?_', mask_path.stem)
            # 构建目标路径
            full_sample_path = base_template /'FullSample_slice'/ patient_id / (matches.group(0)[:-1]+ mask_path.suffix)
            region_path=base_template/'HeartRegion'/patient_id/(matches.group(1)+'_heartregion.npy')
            return full_sample_path,region_path

        file_path = self.samplelist[index][0]
        index=self.samplelist[index][1]
        noise_mask=np.load(file_path)[index]
        
        slice_path,reigion_path=get_full_sample_path(file_path)
        k_grth = np.load(slice_path)
        center=np.load(reigion_path)
        rawk=self.transform(k_grth,center)
        rawk=normalize_complex(rawk)
        raw_k_2=np.stack((np.real(rawk),np.imag(rawk)),axis=0).astype(np.float32)
        max_vals = np.abs(raw_k_2).max(axis=2, keepdims=True).max(axis=3, keepdims=True)
        min_vals = np.abs(raw_k_2).min(axis=2, keepdims=True).min(axis=3, keepdims=True)
        raw_k_2=(raw_k_2-min_vals)/(max_vals-min_vals)

# 对每个t维度进行独立的归一化
        
        crop_k_2=self.gen_crop_k_and_target(raw_k_2,noise_mask)
       
        inputs = torch.from_numpy(crop_k_2)
        rawk=torch.from_numpy(raw_k_2)
        
# 对每个t维度进行独立的归一化

       
        #inputs_normalized=inputs


        noise_target=torch.from_numpy(noise_mask[:,0])
        noise_target= noise_target != 0
        return inputs,rawk,noise_target
    
    def __len__(self):
        return self.sample_num

