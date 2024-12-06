import torchreid
import optuna
import time

def objective(trial):
    height = 256
    width = 128
    batch_size_train = 64
    batch_size_test = 100
    
    # Model setup
    model_name = "osnet_ibn_x1_0"
    loss = "softmax"
    
    # Optimizer setup
    optim = "sgd"  #adam
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 0.000005, 0.0005)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    
    # Scheduler setup
    lr_scheduler = "cosine"
    stepsize = 10
    
    # Engine setup
    max_epoch = 50
    eval_freq = 1  # Evaluate after every epoch
    print_freq = 10
    test_only = False
    save_dir = "log/osnet_ibn_x1_0"
    
    # Data manager for training
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="market1501",
        targets="market1501",
        height=height,
        width=width,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        transforms=["random_flip", "random_crop", "random_patch"]
    )
    
    # Model setup
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss=loss,
        pretrained=True,
    )
    
    model = model.cuda()
    
    # Optimizer setup
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=optim,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    
    # Scheduler setup
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=lr_scheduler,
        stepsize=stepsize,
    )
    
    # Engine setup
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
    )

    best_rank1 = 0 
    for epoch in range(1, max_epoch + 1):
        
        # Run one epoch of training
        engine.run(
            save_dir=save_dir,
            max_epoch=epoch,
            eval_freq=eval_freq,
            print_freq=print_freq,
            test_only=test_only,
        )
        
        # Evaluate the model after each epoch
        test_result = engine.test()
        if isinstance(test_result, (list, tuple)):
            rank1 = test_result[0]  
        else:
            rank1 = test_result
        
        best_rank1 = max(best_rank1, rank1)
        
        trial.report(best_rank1, epoch)  
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_rank1  


def main():
    # Create the Optuna study
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=50)  

    print('Optimization finished.')
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')


if __name__ == '__main__':
    main()
