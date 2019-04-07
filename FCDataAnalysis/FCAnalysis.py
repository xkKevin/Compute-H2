#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


# In[4]:

'''
f = pyedflib.EdfReader(r"D:\data\FCDataAnalysis\zhaoyunfengsz1.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
record = np.zeros((n, f.getNSamples()[0]))
for i in range(n):
    record[i, :] = f.readSignal(i)
'''

# In[5]:
plt.rcParams['font.sans-serif']=['SimHei']  #设置字体为黑米
plt.rcParams['axes.unicode_minus']=False    #设置可以显示负号

#p1与p2是两个三维坐标点
points_center = lambda p1,p2:[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
points_dist = lambda p1,p2:((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**0.5


def h2_modify(x,y,L=7):
    '''
    x,y 代表两个信号
    L代表区间线
    该函数返回每个区间的Q、P值
    '''
    min_x = np.min(x)-0.00000001
    max_x = np.max(x)
    lx = len(x)
    bins=np.linspace(min_x,max_x,L)  
    Q=[]
    P=[] # 6
    k=[] # 5
    f=[]
    Qy=[None]*(L-1)
    for i in range(L-1):
        Qy[i] = []
        
    for xi in range(lx):
        for li in range(L-1):
            if bins[li] < x[xi] <=bins[li+1]:
                Qy[li].append(y[xi])
                break
    
    for i in range(L-1):
        if len(Qy[i]):
            Q.append(np.mean(Qy[i]))
            P.append((bins[i]+bins[i+1])/2) 
            if i:
                k.append((Q[i]-Q[i-1])/(P[i]-P[i-1]))
        else:
            q1,q3=np.percentile(x,[25,75]) 
            condition = (q3-q1)*2  #默认是1.5倍 x-q3 > c or q1 - x > c
            outliers1 = np.where(x>condition+q3)[0]
            outliers2 = np.where(x<q1-condition)[0]
            x =  np.delete(x,np.r_[outliers1,outliers2])
            y =  np.delete(y,np.r_[outliers1,outliers2])
            return h2_modify(x,y,L)

    px=P.copy()
    px[0]=min_x
    px[L-2]=max_x
    
    for xi in x:
        for li in range(L-2):
            if px[li] < xi <= px[li+1]:
                fx=k[li]*(xi-P[li])+Q[li]
                f.append(fx)
                break
  
    #f = np.array(f) 当长度相等，可以不加
    my=np.mean(y)
    sst=np.sum((y-my)**2)
    sse=np.sum((y-f)**2)
    h2=1-sse/sst   
    return h2


# In[6]:


def H2_filter(signals,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    '''
    在H2_whole3方法的基础上增加filter功能
    signals：原始信号，numpy矩阵类型（二维矩阵）
    slideWindows:滑动窗口
    duration：信号的持续时间
    HP，LP：接收归一化截止频率
    '''
    #time_flexity=1
    signals_num = len(signals)
    if signals_num < 2:
        print("Please input at least 2 signals to compute h2!")
        return None,None
    
    h2_num = int((duration-slideWindow-maxlag)//step + 1)
    if h2_num <= 0:
        print("Duration is too short to compute h2!")
        return None,None
    
    if maxlag <= 0:
        print("Time maxlag must be greater than zero!")
        
    b = a = []
    if HP==0 and LP>0: #低通
        b,a = scipy.signal.butter(2,2*LP/sampleFreq,"lowpass")
    elif LP==0 and HP>0: #高通
        b,a = scipy.signal.butter(2,2*HP/sampleFreq,"highpass")
    elif LP>0 and HP>0: #带通
        b,a = scipy.signal.butter(2,[2*HP/sampleFreq,2*LP/sampleFreq],"bandpass")
    
    h2_value = []
    lag_value = []
    
     # 计算每次窗口滑动时的H2（每个时间段）
    for ti in range(h2_num):
        start_time_window = start_time + ti*step
        stop_time_window = start_time_window + slideWindow
        start_samples = int(start_time_window*sampleFreq)
        stop_samples = int(stop_time_window*sampleFreq)
        # 对角线上的值设置为 0
        signals_h2 = np.zeros((signals_num,signals_num))
        signals_lag = np.zeros((signals_num,signals_num))
        # 计算每两个信号之间的H2: (0,1)，(0,2)，...，(0,n-1)；(1,2)，(1,3)，...，(1,n-1)；...(n-2,n-1)
        for si1 in range(signals_num-1):
            s1 = signals[si1][start_samples:stop_samples]
            if len(a):
                s1 = scipy.signal.filtfilt(b,a,s1)
            # [s1] = setSignaltoZero([signals[si1][start_samples:stop_samples]]) # 做归0处理
            for si2 in range(si1+1,signals_num):
                s2 = signals[si2][start_samples:stop_samples]
                if len(a):
                    s2 = scipy.signal.filtfilt(b,a,s2)
                #[s2] = setSignaltoZero([signals[si2][start_samples:stop_samples]])
                # 设置两个信号之间的H2的初始值为0
                two_signals_h2 = np.zeros(2) 
                two_signals_lag = np.zeros(2) 
                
                h2_00 = [h2_modify(s1,s2,L),h2_modify(s2,s1,L)]
                #time_flexity += 1
                two_signals_h2[0] = h2_00[0]
                two_signals_h2[1] = h2_00[1]
                # 考虑时间延迟
                # 计算x到y
                sample_maxlag = int(maxlag*sampleFreq)
                for tagi in range(1,sample_maxlag): # x向右移动，符号为正；y向右移动，符号为负
                    #print(start_samples,stop_samples)
                    #time_flexity += 2
                    s1m = signals[si1][start_samples+tagi:stop_samples+tagi]
                    s2m = signals[si2][start_samples+tagi:stop_samples+tagi]
                    #[s1m,s2m] = setSignaltoZero([s1m,s2m],1)  # 做归0处理
                    
                    h2_mn = [h2_modify(s1m,s2,L),h2_modify(s2,s1m,L)] # s1向右移  H2(s1m,s2,L)
                    h2_nm = [h2_modify(s1,s2m,L),h2_modify(s2m,s1,L)] # s2向右移  H2(s1,s2m,L)
                    
                    if (h2_mn[0]>two_signals_h2[0])&(h2_mn[0]>h2_nm[0]):
                        two_signals_h2[0] = h2_mn[0]
                        two_signals_lag[0] = tagi
                        
                    elif h2_nm[0]>two_signals_h2[0]:
                        two_signals_h2[0] = h2_nm[0]
                        two_signals_lag[0] = -tagi
                        
                    if (h2_mn[1]>two_signals_h2[1])&(h2_mn[1]>h2_nm[0]):
                        two_signals_h2[1] = h2_mn[1]
                        two_signals_lag[1] = -tagi
                        
                    elif h2_nm[1]>two_signals_h2[1]:
                        two_signals_h2[1] = h2_nm[1]
                        two_signals_lag[1] = tagi
                
                signals_h2[si1,si2] = two_signals_h2[0]
                signals_h2[si2,si1] = two_signals_h2[1]
                signals_lag[si1,si2] = two_signals_lag[0]
                signals_lag[si2,si1] = two_signals_lag[1]
                
        h2_value.append(signals_h2)
        lag_value.append(signals_lag)
    
    #print(time_flexity)
    return h2_value,lag_value    

	
def H2_max(h2,lag,s1=0,s2=1):
    '''
    返回这组信号中s1与s2的最终h2（最大值）及其对应的时间延迟
    s1,s2代表数字，如 0,3
    '''
    pnum = len(h2) # 时间点的数量
    h2_max=[None]*pnum
    lag_max=[None]*pnum
    for pi in range(pnum):
        if h2[pi][s1,s2] >= h2[pi][s2,s1]:
            h2_max[pi] = h2[pi][s1,s2]
            lag_max[pi] = lag[pi][s1,s2]
        else:
            h2_max[pi] = h2[pi][s2,s1]
            lag_max[pi] = lag[pi][s2,s1]
    return h2_max,lag_max

	
def h2_median(h2_value):
    sub_median=[]
    for i in range(len(h2_value)):
        sub_median.append(np.median(h2_value[i]))
    return np.median(sub_median)
	

def H2_cod(zone1,zone2,cod1,cod2,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    h2_bwt=[]
    h2_bwt_max=[]
    h2_bwt_median=[]
    codistance = []
    lz2 = len(zone2)
    for i in range(len(zone1)):
        xi = i*lz2
        for j in range(lz2):
            yi = xi+j
            h2_bwt.append(H2_filter([zone1[i],zone2[j]],start_time,duration,sampleFreq,slideWindow,step,maxlag,L,HP,LP))
            h2_bwt_max.append(H2_max(h2_bwt[yi][0],h2_bwt[yi][1]))
            h2_bwt_median.append(np.median(h2_bwt_max[yi][0]))
            codistance.append(points_dist(cod1[i],cod2[j]))
    return h2_bwt,h2_bwt_max,h2_bwt_median,np.median(h2_bwt_median),codistance
	
	
def H2_bwt(zone1,zone2,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    h2_bwt=[]
    h2_bwt_max=[]
    h2_bwt_median=[]
    for i in range(len(zone1)):
        h2_bwt.append(H2_filter([zone1[i],zone2[i]],start_time,duration,sampleFreq,slideWindow,step,maxlag,L,HP,LP))
        h2_bwt_max.append(H2_max(h2_bwt[i][0],h2_bwt[i][1]))
        h2_bwt_median.append(np.median(h2_bwt_max[i][0]))
        
    return h2_bwt,h2_bwt_max,h2_bwt_median,np.median(h2_bwt_median)
	
	
def H2_bwt_cod(zone1,zone2,cod1,cod2,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    h2_bwt=[]
    h2_bwt_max=[]
    h2_bwt_median=[]
    codistance = []
    for i in range(len(zone1)):
        h2_bwt.append(H2_filter([zone1[i],zone2[i]],start_time,duration,sampleFreq,slideWindow,step,maxlag,L,HP,LP))
        h2_bwt_max.append(H2_max(h2_bwt[i][0],h2_bwt[i][1]))
        h2_bwt_median.append(np.median(h2_bwt_max[i][0]))
        codistance.append(points_dist(cod1[i],cod2[i]))
    return h2_bwt,h2_bwt_max,h2_bwt_median,np.median(h2_bwt_median),codistance

	
def h2_in_suit(zone,cod,label,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    h2_bwt=[]
    h2_bwt_max=[]
    h2_bwt_median=[]
    h2_per_max = []
    codistance = []
    lz = len(zone)
    for i in range(lz-1):
        for j in range(i+1,lz):
            h2_bwt.append(H2_filter([zone[i],zone[j]],start_time,duration,sampleFreq,slideWindow,step,maxlag,L,HP,LP))
            h2_bwt_max.append(H2_max(h2_bwt[-1][0],h2_bwt[-1][1]))
            a,b=np.percentile(h2_bwt_max[-1][0],[50,100])
            h2_bwt_median.append(a)
            h2_per_max.append(b)
            codistance.append(points_dist(cod[i],cod[j]))
    line([i[0] for i in h2_bwt_max],np.round(codistance,3),label+' FC随时间变化的折线图',start_time,step)
    violin([i[0] for i in h2_bwt_max],np.round(codistance,3),label+' FC小提琴图')
    for i in range(len(h2_bwt_median)):
        plt.text(i+1,h2_per_max[i]+0.01,np.round(h2_bwt_median[i],3),ha='center',va='bottom',fontsize=15) # ha：水平对齐，va：垂直对齐
    return h2_bwt,h2_bwt_max,h2_bwt_median,np.median(h2_bwt_median),codistance


def h2_btw_suit(zone1,zone2,cod1,cod2,label,start_time,duration,sampleFreq,slideWindow=2,step=1,maxlag=0.1,L=7,HP=0,LP=0):
    h2_bwt=[]
    h2_bwt_max=[]
    h2_bwt_median=[]
    h2_per_max = []
    codistance = []
    lz2 = len(zone2)
    for i in range(len(zone1)):
        for j in range(lz2):
            h2_bwt.append(H2_filter([zone1[i],zone2[j]],start_time,duration,sampleFreq,slideWindow,step,maxlag,L,HP,LP))
            h2_bwt_max.append(H2_max(h2_bwt[-1][0],h2_bwt[-1][1]))
            a,b=np.percentile(h2_bwt_max[-1][0],[50,100])
            h2_bwt_median.append(a)
            h2_per_max.append(b)
            codistance.append(points_dist(cod1[i],cod2[j]))
    line([i[0] for i in h2_bwt_max],np.round(codistance,3),label+' FC随时间变化的折线图',start_time,step)
    violin([i[0] for i in h2_bwt_max],np.round(codistance,3),label+' FC小提琴图')
    for i in range(len(h2_bwt_median)):
        plt.text(i+1,h2_per_max[i]+0.01,np.round(h2_bwt_median[i],3),ha='center',va='bottom',fontsize=15) # ha：水平对齐，va：垂直对齐
    return h2_bwt,h2_bwt_max,h2_bwt_median,np.median(h2_bwt_median),codistance
	

def line(llist,label,title,start,step):
    plt.figure(figsize=(17,5))
    x = list(range(start,step*len(llist[0])+start,step))
    for i in range(len(llist)):
        plt.plot(x,llist[i],label=label[i])
    #plt.xticks(np.linspace(0,ll-1,pd),np.linspace(start,start+ll-1,pd,dtype=int))
    plt.legend()
    plt.title(title,fontsize=18)
    plt.grid()
    plt.show()

def violin(vlist,label,title):
    plt.figure(figsize=(16,7))
    vp = plt.violinplot(vlist,showmedians=True) # data里是元素（如data1）还不允许是Series
    #plt.scatter('01',pz[0])
    ax=plt.gca()
    ax.spines['bottom'].set_position(('data',0)) 
    plt.title(title,fontsize=18)
    #color = ['LightSkyBlue','DeepSkyBlue','Cyan','MediumSpringGreen','GreenYellow','Yellow','Orange','Chocolate','LightPink','Violet']
    color=['DeepSkyBlue','DarkTurquoise','MediumSpringGreen','GreenYellow','Yellow','Orange','SandyBrown','LightPink','Violet','BlueViolet']
    i=0
    for pc in vp['bodies']:
        pc.set_facecolor(color[i%10])
        pc.set_edgecolor('black')
        i+=1
    plt.xticks(range(1,len(label)+1),label)  # 只能这么设置label
    #plt.yticks([i/10 for i in range(0,11)])
    plt.grid()
    #plt.show()
	

def violinbox(vlist,title):
    plt.figure(figsize=(12,7))
    vp = plt.violinplot(vlist,showmedians=True) # data里是元素（如data1）还不允许是Series
    color=['DeepSkyBlue','MediumSpringGreen','Yellow','Orange','Violet','BlueViolet']
    i=0
    for pc in vp['bodies']:
        pc.set_facecolor(color[i])
        pc.set_edgecolor('black')
        i+=1
    plt.boxplot(vlist,widths=0.02,patch_artist=True,boxprops={'color':'k','facecolor':'k'},showcaps=False,\
                 showmeans=True,meanprops={'markerfacecolor':'r'},showfliers=False,medianprops={'color':'k'})
    for i in range(0,6):
        median = np.median(vlist[i])
        plt.text(i+1,np.max(vlist[i])+0.006,np.round(median,3),ha='center',va='bottom',fontsize=15)
    #ax=plt.gca()
    #ax.spines['bottom'].set_position(('data',0)) 
    plt.ylim(0,1)
    plt.title(title,fontsize=18)
    plt.grid()
    plt.xticks(list(range(1,7)),['EZ','PZ','NIZ','EZ-PZ','EZ-NIZ','PZ-NIZ'])  # 只能这么设置label