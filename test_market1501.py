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
        transforms=["random_flip", "random_crop", "random_patch"], 
    )
    
    model = torchreid.models.build_model(
        name="osnet_ibn_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="sgd",
        lr=0.065,
        weight_decay=0.0005,
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="cosine"
    )
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        use_gpu=True,
    )
    
    engine.run(
        save_dir="log\osnet_ibn_x1_0",
        max_epoch=350,
        eval_freq=10,
        print_freq=10,
        test_only=False,
        visrank=False,
    )


if __name__ == '__main__':
    main()
