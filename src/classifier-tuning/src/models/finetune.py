import os
import copy
import time
import tqdm

import torch

import clip.clip as clip

import copy

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, SoftTargetCrossEntropy, SoftTargetCrossEntropy_T, KD_loss, Hard_pseudo_label_CE, cosine_lr_const_warmup, Poly1_cross_entropy

import src.datasets as datasets

from torch.optim.lr_scheduler import _LRScheduler

def freeze_BN(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    return model

class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

class DataReader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)


    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            self.construct_iter()
            return self.dataloader_iter.next()


def finetune_zsl(args, train_data_aug=None):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head

        input_key = 'features'
        if train_data_aug == None:
            preprocess_fn = image_classifier.val_preprocess
        else:
            preprocess_fn = train_data_aug
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        # preprocess_fn = image_classifier.train_preprocess
        preprocess_fn = train_data_aug
        image_enc = None
        image_classifier.process_images = True
        print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    print('Number of params group being optimized: {}'.format(len(params)))
    print('No. of params being optimized: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, eps=1e-8, betas=(0.9, 0.999))
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    print('Zero-shot performance:')
    if args.freeze_encoder:
        zsl_model = ImageClassifier(image_classifier.image_encoder, model.module)
    else:
        zsl_model = model.module

    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, \
        RandomHorizontalFlip
    from PIL import Image

    def _convert_to_rgb(image):
        return image.convert('RGB')

    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    eval_results = evaluate(zsl_model, args)

    # --- soft_label
    if args.sl > 0:
        zsl_teacher = copy.deepcopy(zsl_model)
        if args.freeze_encoder:
            zsl_teacher = zsl_teacher.classification_head   # only for LP
        if not args.hard_pseudo_label:
            loss_sl = SoftTargetCrossEntropy_T(args.sl_T)
        else:
            loss_sl = Hard_pseudo_label_CE()
        zsl_teacher.eval()

    best_acc = 0
    data_loader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=image_enc)
    for epoch in range(args.epochs):
        model.train()
        model = freeze_BN(model)

        for i, batch in enumerate(data_loader):
            start_time = time.time()

            if not next(model.parameters()).is_cuda:
                model = model.cuda()

            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            if args.sl > 0:
                teacher_logits = zsl_teacher(inputs)
                loss_soft = loss_sl(logits, teacher_logits)
                loss_hard = loss_fn(logits, labels)
                loss = args.sl * loss_soft + (1-args.sl) * loss_hard
            else:
                loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None and (epoch == args.epochs - 1):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch + 1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch + 1}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        args.current_epoch = epoch
        # eval_results = evaluate(image_classifier, args)
        # Evaluate
        ds_name = args.eval_datasets[0]
        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            eval_results = evaluate(image_classifier, args)
            val_acc = eval_results[ds_name+':top1']
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(args.save, exist_ok=True)
                model_path = os.path.join(args.save, f'checkpoint_best.pt')
                print('Saving model to', model_path)
                image_classifier.save(model_path)

    print('best acc of {}: {}'.format(args.save, best_acc))

    if args.save is not None:
        return model_path


def finetune_fsl(args, train_data_aug=None):
# def finetune(args, train_data_aug=None, mixup_fn=None, train_data_aug_syn=None):
    '''
    Finetune with fixed BN for syn and real data
    two dataloaders: real and syn
    two loss term and a weight hyper

    Args:
        args:
        train_data_aug:

    Returns:

    '''
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head

        input_key = 'features'
        preprocess_fn = train_data_aug
        preprocess_fn_syn = train_data_aug
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = train_data_aug
        preprocess_fn_syn = train_data_aug
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
        # not tuning vision encoder
        for para in model.image_encoder.parameters():
            para.requires_grad = False


    dataset_class = getattr(datasets, args.train_dataset)
    dataset_real = dataset_class(
        preprocess_fn,
        location=args.data_location_real,
        batch_size=args.batch_size_real
    )
    dataset_syn = dataset_class(
        preprocess_fn_syn,
        location=args.data_location,
        batch_size=args.batch_size_syn
    )

    num_batches_real = len(dataset_real.train_loader)
    num_batches_syn = len(dataset_syn.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    print('Number of params group being optimized: {}'.format(len(params)))
    print('No. of params being optimized: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, eps=1e-8, betas=(0.9, 0.999))
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches_syn)    # we set the iterations depending on syn data

    print('Zero-shot performance:')
    if args.freeze_encoder:
        zsl_model = ImageClassifier(image_classifier.image_encoder, model.module)
    else:
        zsl_model = model.module
    eval_results = evaluate(zsl_model, args)

    # --- soft_label
    if args.sl > 0:
        zsl_teacher = copy.deepcopy(zsl_model)
        if args.freeze_encoder:
            zsl_teacher = zsl_teacher.classification_head   # only for LP
        if not args.hard_pseudo_label:
            loss_sl = SoftTargetCrossEntropy_T(args.sl_T)
        else:
            loss_sl = Hard_pseudo_label_CE()
        zsl_teacher.eval()

    best_acc = 0
    data_loader_real = get_dataloader(dataset_real, is_train=True, args=args, image_encoder=image_enc, is_real=True)
    real_data_reader = DataReader(data_loader_real)
    data_loader_syn = get_dataloader(dataset_syn, is_train=True, args=args, image_encoder=image_enc, is_real=False)

    for epoch in range(args.epochs):
        model.train()
        model = freeze_BN(model)

        real_data_reader.construct_iter()

        for i, batch_syn in enumerate(data_loader_syn):
            start_time = time.time()

            if not next(model.parameters()).is_cuda:
                model = model.cuda()

            step = i + epoch * num_batches_syn
            scheduler(step)
            optimizer.zero_grad()

            # ----- syn data forward -----

            batch = maybe_dictionarize(batch_syn)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            if args.sl > 0:
                # import ipdb
                # ipdb.set_trace(context=20)
                teacher_logits = zsl_teacher(inputs)
                loss_soft = loss_sl(logits, teacher_logits)
                loss_hard = loss_fn(logits, labels)
                loss_syn = args.sl * loss_soft + (1-args.sl) * loss_hard
            else:
                loss_syn = loss_fn(logits, labels)

            # ----- real data forward -----
            batch_real = real_data_reader.read_data()
            batch_real = maybe_dictionarize(batch_real)
            inputs_real = batch_real[input_key].cuda()
            labels_real = batch_real['labels'].cuda()

            logits_real = model(inputs_real)
            loss_real = loss_fn(logits_real, labels_real)

            loss = args.loss_weight_real * loss_real + args.loss_weight * loss_syn

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader_syn)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset_syn.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None and (epoch == args.epochs - 1):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch + 1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch + 1}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        args.current_epoch = epoch
        # eval_results = evaluate(image_classifier, args)
        # Evaluate
        ds_name = args.eval_datasets[0]
        if (args.epochs > 100 and (epoch + 1) % (args.epochs//30) == 0 ) or (args.epochs <100 and (epoch + 1) % (args.epochs//30) == 0 ) or epoch == args.epochs - 1:
            eval_results = evaluate(image_classifier, args)
            val_acc = eval_results[ds_name+':top1']
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(args.save, exist_ok=True)
                model_path = os.path.join(args.save, f'checkpoint_best.pt')
                print('Saving model to', model_path)
                image_classifier.save(model_path)

    print('best acc of {}: {}'.format(args.save, best_acc))

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune_zsl(args)
