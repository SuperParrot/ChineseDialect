import numpy as np
import pandas as pd
import os
import librosa as librosa
import librosa.display
from scipy import interpolate, signal
import scipy.io as scio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.cluster import DBSCAN
from pykalman import KalmanFilter
from tqdm import tqdm
import soundfile

from ssim import SSIM


def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def read_wav(path, sample_rate):
    mpl.rcParams["font.sans-serif"] = ["SimHei"]

    wav_dura = librosa.get_duration(path=path)
    y, sr = librosa.load(path, sr=sample_rate, offset=0.0, duration=wav_dura)
    '''
    x = np.arange(0, wav_dura, 1 / sr)  # 时间刻度
    plt.plot(x, y)
    plt.xlabel('times')  # x轴时间
    plt.ylabel('amplitude')  # y轴振幅
    plt.title(f, fontsize=12, color='black')  # 标题名称、字体大小、颜色
    plt.show()
    '''
    return wav_dura, y

def get_corrcoe(x, y):
    ux = np.mean(x)
    uy = np.mean(y)
    sigx = np.std(x)
    sigy = np.std(y)
    coxy = np.mean(x * y) - ux * uy
    sxy = (coxy + 1e-5) / (sigx * sigy + 1e-5)

    return sxy

def get_simi(x, y):
    ux = np.mean(x)
    uy = np.mean(y)
    sigx = np.std(x)
    sigy = np.std(y)

    diff_1 = np.mean(np.abs(np.diff(x)-np.diff(y)))
    diff_2 = np.mean(np.abs(np.diff(x, n=2)-np.diff(y, n=2)))
    x_range=x.max()-x.min()
    y_range=y.max()-y.min()
    range_diff = np.abs(x_range-y_range)
    u_diff=np.abs(ux-uy)
    mae = np.mean(np.abs(x-y))
    x_corr=get_corrcoe(x, range(len(x)))
    y_corr=get_corrcoe(y, range(len(y)))

    keypoints_diff=[]
    keypoints_diff.append(np.abs(x[0]-y[0]))
    keypoints_diff.append(np.abs(x[-1]-y[-1]))
    keypoints_diff.append(np.abs(x.max()-y.max()))
    keypoints_diff.append(np.abs(x.min()-y.min()))

    simi_val=1.0

    if(x_range<0.8 and y_range<0.8):
        simi_val_tmp=(5.0-u_diff)/5.0
        if(u_diff<0.5):
            simi_val *= simi_val_tmp ** 0.5
        elif(u_diff<0.75):
            simi_val *= simi_val_tmp ** 2.0
        elif(u_diff<1.0):
            simi_val *= simi_val_tmp ** 3.0
        else:
            simi_val *= simi_val_tmp ** 4.0

        simi_val *= ((5.0- mae) / 5.0)
    else:
        if(x_corr * y_corr < 0.0):
            simi_val *= ((2.0 - np.abs(x_corr - y_corr)) / 2.0) ** 2.0
            simi_val *= ((5.0- mae) / 5.0) ** 2.0
        else:
            simi_val *= ((2.0 - np.abs(x_corr - y_corr)) / 2.0)
            simi_val *= ((10.0-diff_1) / 10.0) ** 2.0
            simi_val *= ((20.0-diff_2) / 20.0) ** 4.0
            simi_val *= (5.0 - range_diff) / 5.0
            simi_val *= ((5.0 - mae) / 5.0) ** 0.5

            for keypoint_diff in keypoints_diff:
                if(keypoint_diff<0.9):
                    simi_val *= ((5.0 - keypoint_diff) / 5.0) ** 0.5
                elif (keypoint_diff < 1.1):
                    simi_val *= ((5.0 - keypoint_diff) / 5.0)
                elif(keypoint_diff<1.3):
                    simi_val *= ((5.0 - keypoint_diff) / 5.0) ** 2.0
                elif(keypoint_diff<1.5):
                    simi_val *= ((5.0 - keypoint_diff) / 5.0) ** 3.0
                else:
                    simi_val *= ((5.0 - keypoint_diff) / 5.0) ** 4.0

    return simi_val


def get_distance(x ,y):
    return 1.0-get_simi(x, y)


def get_simiMat(T_all):
    simi_result = np.empty((len(T_all), len(T_all)))
    for i in range(len(T_all)):
        Ti=T_all[i]
        for j in range(len(T_all)):
            Tj = T_all[j]
            # ssim_val=ssim_opt.SSIM1d(Ti, Tj)

            simi_val = get_simi(Ti, Tj)
            simi_result[i, j] = simi_val

    return simi_result


def draw_simiMat(simi_result, label_names=None, title_text=''):
    plt.title(title_text, font={'family': 'SimSun', 'weight': 'semibold', 'size': 18})
    font = {'family': 'SimSun',
            'weight': 'normal',
            'size': 14,
            }

    ax = plt.subplot()
    if (label_names != None):
        plt.xticks(np.arange(simi_result.shape[0]), labels=label_names, rotation=45, rotation_mode="anchor", ha="right",
                   font=font)
        plt.yticks(np.arange(simi_result.shape[0]), labels=label_names, font=font)
    else:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    im = ax.imshow(simi_result, cmap='Reds', norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
    for i in range(simi_result.shape[0]):
        for j in range(simi_result.shape[1]):
            text = plt.text(j, i, '%.2lf' % simi_result[i, j], ha="center", va="center", color="c", font={'size': 12})

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=16)
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    cb.set_label('Similarity', fontdict={'family':'Times New Roman', 'size': 16})
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
    plt.show()

def remove_consonant(sig, lframe, window_name='Hamming'):
    result=[]
    start=0
    max_en=-1

    window=choose_windows(window_name, N=lframe)
    while(start<len(sig)):
        frame=sig[start:min(start+lframe, len(sig))]
        frame_windowed=frame*window[0:len(frame)]
        frame_en=np.abs(frame_windowed).mean()
        #print(frame.shape)
        #print(frame_en)
        if(frame_en>max_en):
            max_en=frame_en

        start+=lframe

    start=0
    started=False
    while (start < len(sig)):
        frame = sig[start:min(start + lframe, len(sig))]
        frame_windowed = frame * window[0:len(frame)]
        # print(frame.shape)
        # print(np.abs(frame_windowed).mean())

        if(not started):
            if (np.abs(frame_windowed).mean() > max_en * 0.6):
                started=True
        if(started):
            result.extend(frame)
            if (np.abs(frame_windowed).mean() < max_en * 0.3):
                break

        start += lframe

    result=np.array(result)
    return result


def get_f0(sig, sample_rate, fmin, fmax, lframe):
    f0_tmp, voiced_flag, voiced_prob = librosa.pyin(sig, sr=sample_rate, fmin=fmin, fmax=fmax, frame_length=lframe)
    # plt.plot(f0_tmp)
    # plt.show()
    f0 = []
    length = len(f0_tmp)
    #print(f0_tmp, voiced_flag, voiced_prob)
    #print(f0_tmp, voiced_flag)
    #print(voiced_flag)
    # exit(0)
    started = False
    false_cnt = 0

    f0_start_tmp=[]
    for i in range(length):
        if(not started):
            if((not np.isnan(f0_tmp[i])) and voiced_flag[i] and voiced_prob[i]>0.3):
                f0_start_tmp.append(f0_tmp[i])
                if(len(f0_start_tmp)>=3):
                    started=True
                    f0 = f0_start_tmp
            else:
                f0_start_tmp.clear()
        else:
            if ((not np.isnan(f0_tmp[i])) and voiced_flag[i] and voiced_prob[i]>0.04):
                f0.append(f0_tmp[i])
                started=True
                false_cnt=0
            else:
                if (started):
                    false_cnt += 1
                    if (false_cnt > 3):
                        break

        '''
        if ((not np.isnan(f0_tmp[i])) and voiced_flag[i]):
            f0.append(f0_tmp[i])
            started = True
            false_cnt = 0
        else:
            if (started):
                false_cnt += 1
                if (false_cnt > 3):
                    break
        '''

    return f0

'''
https://zhuanlan.zhihu.com/p/106582238
'''
def Kalman1D(x, damping=1.0):
    """
    卡尔曼滤波，缺点：耗时较长
    :param x 时间序列
    :damping 协方差，控制参数
    :return x_hat 平滑后的时间序列
    """
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = x[0]
    transition_matrix = 1
    transition_covariance = 0.1
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    x_hat, state_cov = kf.smooth(x)
    return x_hat[:,0]

def algorithm(data_path, lframe_time=0.015):
    sample_rate = 22050

    lframe=int(sample_rate * lframe_time)

    f0_all=[]
    f0_data_all=[]

    fmin=librosa.note_to_hz('C2')
    fmax=librosa.note_to_hz('C7')
    print('fim: %d\tfmax: %d'%(fmin, fmax))

    sos = signal.butter(64, [40, 3400], 'bandpass', fs=sample_rate, output='sos')

    toneType_idx=1
    while(True):
        subpath=data_path + "{:0>2d}".format(toneType_idx) + '/'
        if(not os.path.isdir(subpath)):
            break

        '''
        if(toneType_idx>8):
            break
        '''

        f0_sum=[]
        for file_name in os.listdir(subpath):

            if((not file_name.endswith('.mp3')) and (not file_name.endswith('.wav'))):
                continue

            print('Reading %s...'%file_name)
            wav_dura, audio_sig = read_wav(subpath + file_name, sample_rate)

            '''
            if (not file_name.endswith('.mat')):
                continue

            print('Reading %s...' % file_name)
            input_mat=scio.loadmat(subpath + file_name)
            audio_sig=input_mat['sig_vowelSeg'][0]
            del input_mat
            '''

            audio_sig = signal.sosfilt(sos, audio_sig)
            audio_sig=audio_sig/np.abs(audio_sig).max()
            plt.rcParams['axes.unicode_minus']=False
            '''
            plt.subplot(1,3,1)
            plt.plot(audio_sig)
            plt.ylim(-1.0, 1.0)
            plt.ylabel('Intensity', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            plt.xlabel('Time series', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            '''

            '''
            plt.subplot(1, 3, 2)
            plt.plot(audio_sig)
            plt.ylim(-1.0, 1.0)
            plt.ylabel('Intensity', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            plt.xlabel('Time series', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            '''
            audio_sig=remove_consonant(audio_sig, int(sample_rate * 0.025))

            '''
            plt.subplot(1, 3, 3)
            plt.plot(audio_sig)
            plt.ylim(-1.0, 1.0)
            plt.ylabel('Intensity', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            plt.xlabel('Time series', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            plt.show()
            '''

            #soundfile.write('./tmp1.wav', audio_sig, 22050)
            #exit(0)

            f0=get_f0(audio_sig, sample_rate, fmin, fmax, lframe)
            if(len(f0)<2):
                print('Failed in pYIN.')
                continue

            '''
            plt.subplot(1, 2, 1)
            plt.plot(f0)
            plt.ylim(160.0, 320.0)
            plt.ylabel('Frequency (Hz)', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            plt.xlabel('Time series', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            '''

            f0 = Kalman1D(f0, damping=0.2)
            f=interpolate.interp1d(range(len(f0)), f0, 'linear')
            f0=f(np.linspace(0, len(f0)-1, 12))
            f0=signal.medfilt(f0, 3)
            #print(f0)

            f0_data_all.append(f0)

            #print(f0)

            #plt.subplot(1,2,2)
            #plt.plot(f0)
            #plt.ylim(160.0, 320.0)
            #plt.ylabel('Frequency (Hz)', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            #plt.xlabel('Time series', font={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
            #plt.show()

            f0_sum.append(f0)


        f0_sum=np.array(f0_sum)
        f0_avg=np.zeros(f0_sum.shape[1])
        for idx in range(f0_sum.shape[1]):
            f0_samples=f0_sum[:,idx]
            f0_samples_mean=f0_samples.mean()
            f0_samples_std=f0_samples.std()

            f0_samples_crop=[]
            for i in range(len(f0_samples)):
                if(np.abs(f0_samples[i]-f0_samples_mean)<=f0_samples_std):
                    f0_samples_crop.append(f0_samples[i])
            del f0_samples
            f0_samples_crop=np.array(f0_samples_crop)
            f0_avg[idx]=np.mean(f0_samples_crop)
        del f0_sum, f0_samples_crop

        print(f0_avg)
        #plt.plot(f0_avg)
        #plt.show()
        f0_avg=Kalman1D(f0_avg, damping=0.2)
        #plt.plot(f0_avg)
        #plt.show()

        f0_all.append(f0_avg)

        toneType_idx+=1

    f0_all = np.array(f0_all)
    #print(f0_all.shape)
    #f0_data_all = np.array(f0_data_all)
    a=f0_all.max()
    b=f0_all.min()

    print(a, b)
    lga, lgb=np.log10(a), np.log10(b)

    #print(f0_all)
    T_all=(np.log10(np.array(f0_all))-lgb)/(lga-lgb)*5.0
    for i in range(T_all.shape[0]):
        if(T_all[i].max()>4.5):
            if(T_all[i].min()>4.5):
                T_all[i]=np.ones_like(T_all[i])*4.5
            else:
                T_all[i]=(T_all[i] - T_all[i].min()) / (T_all[i].max() - T_all[i].min()) * (4.5 - T_all[i].min()) + T_all[i].min()

        if (T_all[i].min() < 0.5):
            if(T_all[i].max() < 0.5):
                T_all[i]=np.ones_like(T_all[i])*0.5
            else:
                T_all[i] = (T_all[i] - T_all[i].min()) / (T_all[i].max() - T_all[i].min()) * (T_all[i].max() - 0.5) + 0.5

    tone_names = ['全清平', '次清平', '次浊平', '全浊平', '全清上', '次清上', '次浊上', '全浊上',
                  '全清去', '次清去', '次浊去', '全浊去', '全清入', '次清入', '次浊入', '全浊入']
    plt.suptitle('调值曲线', font={'family': 'SimHei', 'weight': 'semibold', 'size': 20})
    for i in range(len(T_all)):
        plt.subplot(4, 4, i + 1)
        plt.plot(T_all[i])
        plt.ylim(0, 5)
        plt.yticks(np.linspace(0, 5, 6), fontsize=14)
        plt.grid(axis='y')
        plt.title(tone_names[i], font={'family': 'SimSun', 'weight': 'semibold', 'size': 16})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.18, hspace=0.35)
    plt.show()

    #T_data_all = (np.log10(np.array(f0_data_all)) - lgb) / (lga - lgb) * 5.0

    simi_result = get_simiMat(T_all)
    print(simi_result)
    #draw_simiMat(simi_result, title_text='声调相似度矩阵（聚类前）')
    draw_simiMat(simi_result, tone_names, title_text='声调相似度矩阵（聚类前）')

    eps_min=0.0
    eps_max=1.0
    for iter_num in range(100):
        # print(f0_data_all.shape)
        eps_mid = (eps_max + eps_min) / 2
        #print(eps_max, eps_min, eps_mid)
        clustering = DBSCAN(min_samples=2, eps=eps_mid, metric=get_distance).fit(T_all)
        clustered_labels = clustering.labels_
        del clustering
        print(clustered_labels)

        if(np.int32(clustered_labels==-1).sum()==0):
            eps_max=eps_mid
        else:
            eps_min=eps_mid

    f0_merged=np.empty((max(clustered_labels)+1, T_all.shape[1]))
    #T_merged=np.empty((max(clustered_labels)+1, T_data_all.shape[1]))
    for i in range(f0_merged.shape[0]):
        f0_new=f0_all[np.where(clustered_labels==i)]
        #T_new=T_data_all[np.where(clustered_labels==i)]
        f0_new=np.mean(f0_new, axis=0)
        f0_merged[i]=f0_new

    for i in range(f0_merged.shape[0]):
        f0_merged[i]=Kalman1D(f0_merged[i], damping=0.2)

    a = f0_merged.max()
    b = f0_merged.min()
    print(a, b)
    lga, lgb = np.log10(a), np.log10(b)
    T_merged=(np.log10(f0_merged)-lgb)/(lga-lgb)*5.0

    clustering = DBSCAN(min_samples=2, eps=0.3, metric=get_distance).fit(T_merged)
    clustered_labels = clustering.labels_
    del clustering
    print(clustered_labels)

    #print(T_merged.shape)
    #T_merged=np.concatenate((T_merged, T_all[np.where(clustered_labels==-1)]), axis=0)
    #print(T_merged.shape)

    simi_result = get_simiMat(T_merged)
    draw_simiMat(simi_result=simi_result, title_text='声调相似度矩阵（聚类后）')

    if (T_merged.shape[0] == 3):
        plt.figure(figsize=(9, 3))
    elif (T_merged.shape[0] == 4):
        plt.figure(figsize=(12, 3))
    elif (T_merged.shape[0] <= 6):
        plt.figure(figsize=(12, 8))
    elif (T_merged.shape[0] <= 8):
        plt.figure(figsize=(12, 8))
    else:
        plt.figure(figsize=(12, 8))

    plt.suptitle('合并后的调值曲线', font={'family': 'SimHei', 'weight': 'normal', 'size': 18})
    for i in range(T_merged.shape[0]):
        if(T_merged.shape[0]==3):
            plt.subplot(1, 3, i + 1)
        elif(T_merged.shape[0]==4):
            plt.subplot(1,4,i+1)
        elif(T_merged.shape[0]<=6):
            plt.subplot(2, 3, i+1)
        elif (T_merged.shape[0] <= 8):
            plt.subplot(2, 4, i + 1)
        else:
            plt.subplot(2, 5, i + 1)

        plt.plot(T_merged[i])
        plt.ylim(0, 5)
        plt.yticks(np.linspace(0, 5, 6), fontsize=16)
        plt.grid(axis='y')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.14, hspace=0.15)
    plt.show()


