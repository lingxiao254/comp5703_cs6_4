
# coding: utf-8

# In[79]:


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


# In[80]:


# Calculate mse_mae loss

def mix_loss(input, target, mask=None):
    '''
    :param input:
    :param target:
    :param mask:
    :return:
    '''
    balancing_weights = (1, 1, 2, 5, 10, 30)
    # 初始化weights
    weights = torch.ones_like(torch.empty(256,256,3) * balancing_weights[0])
    
    # 对降水量（mm/h）进行数值划分，
    thresholds = [rainfall_to_pixel(ele) for ele in [0.5, 2, 5, 10, 30]] # eg. [0.5, 2, 5, 10, 30]
    for i, threshold in enumerate(thresholds):
        # 小于2,加权1；大于2，加权（1+1）；大于5，加权（1+3）；大于10,加权（1+5）；大于30,加权（1+20）
        weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold)#.float()

    # mask是利用马氏降噪后的掩码
    if mask is not None:
        weights = weights * mask.float()

    # input: S*B*1*H*W
    # error: S*B
    #mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
    #mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
    
    mse = weights * (input-target)**2
    mae = weights * (np.abs((input-target)))        
        
        
    S, B, K = mse.size()
        # 按照seq进行加权,初始设置为等差数列
    w = torch.arange(1.0, 1.0 + S * 1, 1)
    if torch.cuda.is_available():
        w = w.to(mse.get_device())
        # 输出误差的形状Seq*Batch_size
        mse = (w * mse.permute(1, 0)).permute(1, 0)
        mae = (w * mae.permute(1, 0)).permute(1, 0)
    # loss尺度缩放系数5e-5，mae和mse的权重为1：1
    return 0.00005 * (1.0*torch.mean(mse) + 1.0*torch.mean(mae))


# In[81]:


from scipy.misc import imread

def image_read(file_path):
    image = imread(file_path)
    return image


# In[99]:


def read_directory(directory_name):
    
    for filename in os.listdir(r"./"+directory_name):
        img = image_read(directory_name + "/" + filename)
        images.append(img)
    return images


# In[100]:


read_directory("RAD_184392561770451")


# In[121]:


losses = []
for i in range(21, len(images)):
    loss = mix_loss(images[20], images[i])
    losses.append(loss)


# In[123]:


losses


# In[122]:


np.sum(np.array(losses))

