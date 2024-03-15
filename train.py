import time

def train(DEVICE=0, batch_size=4, steps_per_epoch=160, n_filters=32, 
          pooling_steps=5, layers=3, swap_val=False, combine_val=True, 
          ch_bg=True, ch_org=True, ch_diff=True, position=0.5, transfer=False,
          project="intercrackchannels",quantiles=False):
    
    import comet_ml
    experiment = comet_ml.Experiment(
        api_key="",
        project_name=project,
        workspace="",
        display_summary_level=0,
        auto_metric_step_rate=1
        )
    
    params = {
        "batch_size": batch_size,
        "n_filters": n_filters,
        "layers": layers,
        "pooling_steps": pooling_steps,
        "swap_val": swap_val,
        "combine_val": combine_val,
        "position": position,
        "ch_org": ch_org,
        "ch_diff": ch_diff,
        "ch_bg": ch_bg,
        "transfer": transfer
    }

    experiment.log_parameters(params)
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    import numpy as np
    if tf.__version__ != '2.10.0':
        import model as mo
    import fine_tune

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    
    CHANNELS = ch_bg + ch_org + ch_diff
    
    examples, labels, backgrounds = zip(*[fine_tune._load_data(i) for i in range(1,2,1)])
    examples_val, labels_val = zip(*[fine_tune._load_val(i) for i in range(1,2,1)])
    examples_test, labels_test, backgrounds_test = fine_tune._load_test('data/Test')

    if transfer:
        from sklearn.utils import shuffle
        tf_x, tf_y, tf_x_val, tf_y_val = fine_tune._load_transfer('data/Labels/')
        tf_x, tf_y = shuffle(tf_x, tf_y, random_state=0)
        tf_x_val, tf_y_val = shuffle(tf_x_val, tf_y_val, random_state=0)        
        
    examples, labels, examples_val, labels_val, backgrounds  = list(examples), list(labels), list(examples_val), list(labels_val), list(backgrounds)
    labels[0] = labels_test[-1]
    if not quantiles:
        examples_val[0] = [examples_test[int(len(examples_test) * position)]]
        labels_val[0] = labels_test[int(len(examples_test) * position)]
    else:
        quantiles = np.arange(0,1,1/(position+1))[1:]
        examples_val = [[examples_test[int(len(examples_test) * q)]] for q in quantiles]
        labels_val = [labels_test[int(len(examples_test) * q)] for q in quantiles]
        backgrounds += backgrounds * position 
    
    if swap_val:
        examples, labels, examples_val, labels_val = examples_val, labels_val, examples, labels
        
    if combine_val:
        examples, labels, examples_val, labels_val, backgrounds = examples+examples_val, labels+labels_val, examples+examples_val, labels+labels_val, backgrounds+backgrounds
    
    def generator():
        while True:
            X, y = fine_tune._generator(examples_val, backgrounds, labels_val, ch_org=ch_org, ch_diff=ch_diff, ch_bg=ch_bg)
            X, y = fine_tune.random_jitter(X, y, train=True)     
            yield X, y

    def generator_val():
        while True:
            X, y = fine_tune._generator(examples_val, backgrounds, labels_val, ch_org=ch_org, ch_diff=ch_diff, ch_bg=ch_bg)
            X, y = fine_tune.random_jitter(X, y, train=False)     
            yield X, y 
            
    def generator_test():
        for X, y, bg in zip(examples_test, labels_test, backgrounds_test):
            X = fine_tune.test_generator(X, bg, ch_org=ch_org, ch_diff=ch_diff, ch_bg=ch_bg)
            yield X, y 
            
    def generator_transfer():
        for X, y in zip(tf_x, tf_y):
            X, y = fine_tune.random_jitter(X, y, train=True, transfer=True)     
            yield X, y
            
    def generator_transfer_val():
        for X, y in zip(tf_x_val, tf_y_val):
            X, y = fine_tune.random_jitter(X, y, train=True, transfer=True)     
            yield X, y

    def to_dataset(generator):
        ds = tf.data.Dataset.from_generator(generator=generator,
            output_signature=(tf.TensorSpec(shape=(256,512,CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(256,512,1), dtype=tf.float32))).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    trainDS = to_dataset(generator)
    valDS = to_dataset(generator_val)
    testDS = to_dataset(generator_test)
    if transfer:
        transferDS = to_dataset(generator_transfer)
        transferDS_val = to_dataset(generator_transfer_val)

    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            # self.j = 0

        def on_epoch_end(self, epoch, logs=None):    
            metric = self.model.evaluate(testDS, return_dict=True, verbose=0)   
            experiment.log_metrics(metric, prefix='test', epoch=epoch)
            # if epoch % 5 == 0:
            #     for i, (_x, _y) in enumerate(zip(X_test, Y_test)):
            #         metric = self.model.evaluate(_x.reshape(1,256,512,3), _y.reshape(1,256,512,1), return_dict=True, verbose=0)        
            #         experiment.log_metrics(metric, prefix=f'test_X_{epoch}', step=i+self.j)
            #         experiment.log_metrics(metric, prefix='test_X', step=i+self.j, epoch=epoch)
            #     self.j += i + 5

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_PR',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=0,
            min_delta=0.005),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_PR',
            mode='max',
            factor=0.3,
            patience=3,
            verbose=0,
            min_delta=0.001),
        MyCallback()
    ]

    model = mo.build_model(shape=(256,512,CHANNELS), n_filters=n_filters, 
                           layers=layers, pooling_steps=pooling_steps)
    
    if transfer:
        model.fit(transferDS, validation_data=transferDS_val, epochs=8,
                  steps_per_epoch=len(tf_x)//32, validation_steps=len(tf_x)//32,
                    workers=8, use_multiprocessing=True, verbose=0,
                    callbacks=callbacks)
        
        metric = model.evaluate(testDS, return_dict=True, verbose=0)   
        experiment.log_metrics(metric, prefix='semi_final')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                    loss='binary_focal_crossentropy', 
                    metrics=[tf.keras.metrics.AUC(200,'ROC',name='ROC'),
                            tf.keras.metrics.AUC(200,'PR',name='PR')])
    
    model.fit(trainDS.repeat(), validation_data=valDS.repeat(), epochs=100, verbose=0,
                workers=8, use_multiprocessing=True, callbacks=callbacks,
                steps_per_epoch=steps_per_epoch//batch_size, 
                validation_steps=steps_per_epoch//batch_size,)

    metric = model.evaluate(testDS, return_dict=True, verbose=0)   
    experiment.log_metrics(metric, prefix='final')
    for i, (_x, _y) in enumerate(generator_test()):
        metric = model.evaluate(_x[np.newaxis, :], _y[np.newaxis, :], return_dict=True, verbose=0)
        
        if i % 10 == 0:
            prediction = model.predict(_x[np.newaxis, :], verbose=0)
            experiment.log_image(np.vstack([prediction[0][:250,:400],_x[:250,:400,0][:,:,np.newaxis],_y[:250,:400]]), step=i, image_colormap='grey')

        experiment.log_metrics(metric, prefix='individual', step=i)

    prediction = model.predict(_x[np.newaxis, :], verbose=0)
    experiment.log_image(np.vstack([prediction[0][:250,:400],_x[:250,:400,0][:,:,np.newaxis],_y[:250,:400]]), step=i+1, image_colormap='grey')

    print('\n lets sleep \n')
    time.sleep(20)
    
if __name__ == "__main__":
    train()