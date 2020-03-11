import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
from models import BayesianCNN

# def compute_weight_decay(pretrain_loader, valid_loader, args, writers):
#     """
#     From the paper: "All models are trained on the MNIST dataset with a (random
#     but balanced) initial training set of 20 data points, and a validation set
#     of 100 points on which we optimise the weight decay."
#     """
#     corrects = []
#     writer1, writer2 = writers
#     valid_set_size = len(valid_loader) * valid_loader.batch_size
#     weight_decays = [1, 1e-1, 1e-2, 1e-3, 1e-4]
#     for weight_decay in weight_decays:
#         logging.info('Pretraining with weight decay {}'.format(weight_decay))
#         max_correct = 0
#         model = BayesianCNN()
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay) # NB Gal doesn't mention which optimizer they use
#         for epoch in range(50):
#             pretrain_loss = train(args, model, pretrain_loader, optimizer)
#             writer1.add_scalar('pretrain_loss_{}'.format(weight_decay), pretrain_loss, epoch)
#             # test on validation set
#             valid_loss, correct = test(args, model, valid_loader)
#             writer2.add_scalar('pretrain_loss_{}'.format(weight_decay), valid_loss, epoch)
#             writer2.add_scalar('pretrain_correct_{}'.format(weight_decay), correct, epoch)
#             max_correct = max(correct, max_correct)
#         # record max correct
#         corrects.append(max_correct)
#         logging.info('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#                      max_correct, valid_set_size,
#                      100. * max_correct / valid_set_size))
#     return weight_decays[np.argmax(np.array(corrects))]