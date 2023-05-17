import argparse
import os

import torch
import torchvision

from tangle.lab import LabTransactionStore
from tangle.core import Node
from reconstruction import inversefed
from models.femnist.cnn_torch import ClientModel


def main(args):
    tx_store = LabTransactionStore(f'{args.tangle_path}/tangle_data')
    tangle = tx_store.load_tangle(args.round_num)
    learning_rate = get_lr_from_logs(args.tangle_path)
    print(f'Read learning rate of {learning_rate}')
    for tip in tangle.tips:
        tip_tx = tangle.transactions[tip]
        base_model_weights = Node.average_model_params(*[tx_store.load_transaction_weights(parent)
                                                         for parent in tip_tx.parents])
        trained_model_weights = tx_store.load_transaction_weights(tip)
        grads = (trained_model_weights - base_model_weights) / learning_rate
        grads = [torch.Tensor(grad.transpose()) for grad in grads]

        # Load model
        model = ClientModel(62)
        state_dict = {name: torch.Tensor(weight.transpose()) for name, weight in zip(model.state_dict().keys(), trained_model_weights)}
        model.load_state_dict(state_dict)

        # Perform reconstruction
        reconstruct_data(model, grads, tip)


def get_lr_from_logs(path):
    if os.path.exists(f'{path}/1_setup.log'):
        with open(f'{path}/1_setup.log', 'r') as f:
            for line in f:
                if 'learning_rate' in line:
                    start_pos = line.find(':') + 2
                    end_pos = line.find(',')
                    return float(line[start_pos:end_pos])
    return 0.01


def reconstruct_data(model, gradient, tip_id):
    # Reconstruct data!
    dm = 0.9630924688931325
    ds = 0.1611517408001011
    num_images = 10
    config = dict(signed=False,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=1,
                  max_iterations=200,
                  total_variation=1e-1,
                  init='randn',
                  filter='none',
                  lr_decay=False,
                  scoring_choice='loss')

    image_base_path = f'./reconstructed_images/{tip_id}'
    os.makedirs(image_base_path, exist_ok=True)

    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config=config, num_images=num_images, loss_fn='CE')
    output, stats, all_outputs = rec_machine.reconstruct(gradient, labels=None, img_shape=[1, 28, 28], base_path=image_base_path)

    # save the best reconstructed image
    output_denormalized = torch.clamp(output * ds + dm, 0, 1)
    #gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
    for img_idx in range(num_images):
        rec_filename = f'rec_img_idx{img_idx}.png'
        torchvision.utils.save_image(output_denormalized[img_idx], os.path.join(image_base_path, rec_filename))
        #gt_filename = f'{args.name}_gt_img_idx{img_idx}.png'
        #torchvision.utils.save_image(gt_denormalized[img_idx], os.path.join(image_base_path, gt_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='tangle_path',
                        required=True, help='The path to the experiment directory which holds the "tangle_data" dir.')
    parser.add_argument('-r', '--round', dest='round_num',
                        required=True, help='The global round to load the tangle for.')

    args = parser.parse_args()
    main(args)
