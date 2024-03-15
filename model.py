import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPool2D, Concatenate, Add, Conv2DTranspose, GaussianNoise, GroupNormalization

def _conv2d_block(input_tensor, n_filters, i, groupnorm=True, batchnorm=False, concat=False, scale_gn=False, gnoise_everywhere=False, gaussian_noise_std=0., kernel_size = 3, gn_factor=32):
    if gnoise_everywhere:
        input_tensor = GaussianNoise(gaussian_noise_std)(input_tensor)
    
    if groupnorm:
        if scale_gn:
            groups = int(input_tensor.shape[-1]//gn_factor*8) if int(input_tensor.shape[-1]//gn_factor*8) > 0 else int(1)
            x = GroupNormalization(groups)(input_tensor)
        else:
            x = GroupNormalization(max(1,int(n_filters/16)))(input_tensor)
        x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
    elif batchnorm:
        x = BatchNormalization()(input_tensor)
        x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
    else:
        x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)
    if groupnorm:
        if scale_gn:
            groups = int(n_filters//gn_factor*8) if int(n_filters//gn_factor*8) > 0 else int(1)
            x = GroupNormalization(groups)(x)
        else:
            x = GroupNormalization(max(1,int(n_filters/16)))(x)
    elif batchnorm:
        x = BatchNormalization()(x)        
    
    if gnoise_everywhere:
        x = GaussianNoise(gaussian_noise_std)(x)
    x = Conv2D(filters = n_filters, kernel_size = kernel_size, padding = 'same', activation = 'relu')(x)
    if groupnorm:
        if scale_gn:
            x = GroupNormalization(groups)(x)
        else:
            x = GroupNormalization(max(1,int(n_filters/16)))(x)
    elif batchnorm:
        x = BatchNormalization()(x)
    
    if gnoise_everywhere:
        x = GaussianNoise(gaussian_noise_std)(x)
    x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
    
    if groupnorm:
        if scale_gn:
            x = GroupNormalization(groups)(x)
        else:
            x = GroupNormalization(max(1,int(n_filters/16)))(x)
    elif batchnorm:
        x = BatchNormalization()(x)
    if i == 0:
        skip = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)
    else:
        skip = input_tensor

    if concat:
        return Concatenate()([x, skip])
    else:
        return Add()([x, skip])

def build_model(shape, pooling_steps=2, n_filters=16, layers=2, groupnorm=True, concat=False, batchnorm=False,
                loss='binary_focal_crossentropy', lr=0.0005, scale_gn=False, gaussian_noise=False, 
                gaussian_noise_std=0., noise_after=False, gnoise_everywhere=False): 
    pooling_steps = min(pooling_steps, 6)
    encode_layer, decode_layer, bottleneck_layer = layers, layers, layers
    input_data = Input(shape=shape) 
    # print('Required amount of scales: ', 2**pooling_steps, shape[-2])   
    gaussian_noise_std = gaussian_noise_std/2 if gnoise_everywhere else gaussian_noise_std
    if gaussian_noise and not noise_after:
        input_data = GaussianNoise(gaussian_noise_std)(input_data)
    if groupnorm:
        input_data = GroupNormalization(1)(input_data)
    elif batchnorm:
        input_data = BatchNormalization()(input_data)
    c1 = Conv2D(n_filters, kernel_size = 3, padding="same", activation='relu')(input_data)
    if noise_after and gaussian_noise:
        c1 = GaussianNoise(gaussian_noise_std)(c1)
    gn_num = n_filters
    
    for i in range(encode_layer):
        c1 = _conv2d_block(c1, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
    n_filters *= 2
    c2 = MaxPool2D()(c1) if shape[-2] > 2**0 else MaxPool2D(pool_size=2)(c1)
    c7 = c2 if pooling_steps == 1 else None

    if pooling_steps >= 2:
        for i in range(encode_layer):
            c2 = _conv2d_block(c2, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters *= 2
        c3 = MaxPool2D()(c2) if shape[-2] > 2**1 else MaxPool2D(pool_size=2)(c2)
        c7 = c3 if pooling_steps == 2 else None

    if pooling_steps >= 3:
        for i in range(encode_layer):
            c3 = _conv2d_block(c3, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters *= 2
        c4 = MaxPool2D()(c3) if shape[-2] > 2**2 else MaxPool2D(pool_size=2)(c3)
        c7 = c4 if pooling_steps == 3 else None

    if pooling_steps >= 4:
        for i in range(encode_layer):
            c4 = _conv2d_block(c4, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters *= 2
        c5 = MaxPool2D()(c4) if shape[-2] > 2**3 else MaxPool2D(pool_size=2)(c4)
        c7 = c5 if pooling_steps == 4 else None
    
    if pooling_steps >= 5:
        for i in range(encode_layer):
            c5 = _conv2d_block(c5, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters *= 2
        c6 = MaxPool2D()(c5) if shape[-2] > 2**4 else MaxPool2D(pool_size=2)(c5)
        c7 = c6 if pooling_steps == 5 else None
    
    if pooling_steps == 6:
        for i in range(encode_layer):
            c6 = _conv2d_block(c6, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters *= 2
        c7 = MaxPool2D()(c6) if shape[-2] > 2**5 else MaxPool2D(pool_size=2)(c6)
    
    for i in range(bottleneck_layer):
        c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
    n_filters /= 2
    
    if pooling_steps >= 6:
        c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
        c6 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c6)
        c7 = Concatenate()([c7, c6])
        for i in range(decode_layer):
            c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters /= 2
    
    if pooling_steps >= 5:
        c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
        c5 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c5)
        c7 = Concatenate()([c7, c5])
        for i in range(decode_layer):
            c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters /= 2
    
    if pooling_steps >= 4:
        c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
        c4 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c4)
        c7 = Concatenate()([c7, c4])
        for i in range(decode_layer):
            c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters /= 2
    
    if pooling_steps >= 3:
        c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
        c3 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c3)
        c7 = Concatenate()([c7, c3])
        for i in range(decode_layer):
            c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters /= 2

    if pooling_steps >= 2:
        c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
        c2 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c2)
        c7 = Concatenate()([c7, c2])
        for i in range(decode_layer):
            c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)
        n_filters /= 2
    
    c7 = Conv2DTranspose(n_filters, kernel_size = 3, strides = 2, padding = 'same')(c7)
    c1 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c1)
    c7 = Concatenate()([c7, c1])
    for i in range(decode_layer):
        c7 = _conv2d_block(c7, n_filters, i, groupnorm, batchnorm, concat, scale_gn, gnoise_everywhere, gaussian_noise_std)

    if groupnorm:
        if scale_gn:
            c7 = GroupNormalization(int(n_filters//gn_num*8) if int(n_filters//gn_num*8) > 0 else int(1))(c7)
        else:
            c7 = GroupNormalization(max(1,int(n_filters/16)))(c7)
    elif batchnorm:
        c7 = BatchNormalization()(c7)
    
    c8 = Conv2D(n_filters, kernel_size = 3, padding="same", activation='relu')(c7)
    c8 = Conv2D(n_filters, kernel_size = 3, padding="same", activation='relu')(c8)
    outputs = Conv2D(1, kernel_size = 3, padding="same", activation='sigmoid')(c8)

    model = tf.keras.models.Model(inputs=[input_data], outputs=[outputs])
    model.compile(
        loss = loss, 
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr), 
        metrics = [tf.keras.metrics.AUC(200,'ROC',name='ROC'),
                   tf.keras.metrics.AUC(200,'PR',name='PR'),])

    return model
