import torchreid

def main():

    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=64,
        batch_size_test=100,
        transforms=["random_flip", "random_crop", "random_patch"]
    )

    model = torchreid.models.build_model(
        name="osnet_ibn_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        # optim="adam",
        optim="sgd",
        lr=0.001,
        weight_decay=0.0005,
        momentum=0.90 # 0.85 - 0.99
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="cosine", # Use cosine annealing
        stepsize=10
        # lr_scheduler='multi_step', 
        # stepsize=[30, 50, 55]
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        # early_stopping=True, # add early stoping
        # target_metric ="test_acc" # monitor train loss
    )

    # Evaluate on Test data using trained weight
    # weight_path = "log/osnet_ibn_x1_0/model/model.pth.tar-60"
    # torchreid.utils.load_pretrained_weights(model, weight_path)

    engine.run(
        save_dir="log/osnet_ibn_x1_0",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False,
        # test_only=True,
        # visrank=True,
        #normalize_feature=True
    )

if __name__ == '__main__':
    main()