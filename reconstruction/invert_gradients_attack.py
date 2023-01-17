from reconstruction import inversefed

# ATTACK SETTINGS
restarts = 3
max_iterations = 20000
# dummy image initialization mode
# 'randn', 'rand', 'zeros', 'xray', 'mean_xray'
init = 'randn'
# total variation value for cosine similarity loss
# 1e-4 for smaller networks, 1e-1 for larger
tv = 1e-2
# if True, optimize on signed gradients
set_signed = True
# learning rate for attack optimizer
attack_lr = 0.1

# MODEL LOSS FUNCTION, takes care of itself
loss_name = 'CE'
if label_encoding == 'multi':
    img_label = torch.Tensor([img_label]).float()
    img_label = img_label.view(args.num_images,num_classes)
    loss_name = 'BCE'


def compute_model_update(model_weights, parent_models):
    pass


def reconstruct_samples(model_update, input_batch, n_reconstructions):
    # Reconstruct data!
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images,
                                                   loss_fn=loss_name)
    output, stats, all_outputs = rec_machine.reconstruct(input_gradient, labels=labels, img_shape=img_shape,
                                                         dryrun=args.dryrun, set_eval=set_eval)


    # Compute stats
    factor=1/ds # for PSNR computation

    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    # feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    feat_mse = np.nan # placeholder so no errors occur
    test_psnr = inversefed.metrics.psnr(output.detach(), ground_truth, factor=factor)

    if args.save_image and not args.dryrun:

        # save the best reconstructed image
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        for img_idx in range(args.num_images):
            rec_filename = f'{args.name}_rec_img_exp{stats["best_exp"]}_idx{img_idx}.png'
            torchvision.utils.save_image(output_denormalized[img_idx], os.path.join(args.image_path, rec_filename))
            gt_filename = f'{args.name}_gt_img_idx{img_idx}.png'
            torchvision.utils.save_image(gt_denormalized[img_idx], os.path.join(args.image_path, gt_filename))

        # save images of all experiments
        os.makedirs('./images_all/', exist_ok=True)
        output_denormalized = torch.clamp(all_outputs * ds + dm, 0, 1)
        for trial in range(restarts):
            for img_idx in range(args.num_images):
                rec_filename = f'{args.name}_rec_img_exp{trial}_idx{img_idx}.png'
                torchvision.utils.save_image(output_denormalized[trial][img_idx], os.path.join('./images_all/', rec_filename))
    else:
        rec_filename = None
        gt_filename = None

    # Save stats
    for trial in rec_machine.exp_stats:
        all_mses, all_psnrs = [], []
        for img_hist in trial['history']:
            mses = [((rec_img - gt_img).pow(2).mean().item()) for rec_img, gt_img in zip(img_hist, ground_truth)]
            psnrs = [(inversefed.metrics.psnr(rec_img.unsqueeze(0), gt_img.unsqueeze(0), factor=factor)) for rec_img, gt_img in zip(img_hist, ground_truth)]
            all_mses.append(mses)
            all_psnrs.append(psnrs)
        all_metrics = [trial['idx'], trial['rec_loss'], all_mses, all_psnrs]
        with open(f'trial_histories/{args.name}_{trial["name"]}.csv', 'w') as f:
            header = ['iteration', 'loss', 'mse', 'psnr']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*all_metrics))

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} |")

    # Save parameters in table
    # some values are not recorded (e.g., feat_mse, val_acc)
    inversefed.utils.save_to_table(args.table_path, name=f'exp_{args.name}', dryrun=args.dryrun,

                                   model=args.model,
                                   dataset=args.dataset,
                                   trained=args.trained_model,
                                   accumulation=args.accumulation,
                                   restarts=config['restarts'],
                                   OPTIM=args.optim,
                                   cost_fn=args.cost_fn,
                                   indices=args.indices,
                                   weights=args.weights,
                                   scoring=args.scoring_choice,
                                   init=config['init'],
                                   tv=tv,

                                   rec_loss=stats["opt"],
                                   best_idx=stats["best_exp"],
                                   psnr=test_psnr,
                                   test_mse=test_mse,
                                   feat_mse=feat_mse,

                                   target_id=-1,
                                   seed=model_seed,
                                   timing=str(datetime.timedelta(seconds=time.time() - start_time)),
                                   dtype=setup['dtype'],
                                   epochs=args.epochs,
                                   val_acc=None,
                                   rec_img=rec_filename,
                                   gt_img=gt_filename
                                   )


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')