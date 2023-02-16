import os

from src.models.finetune import finetune_fsl
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from PIL import Image

def _convert_to_rgb(image):
    return image.convert('RGB')

normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def classifier_tuning(args):
    assert args.save is not None, 'Please provide a path to store models'
    print('import success')

    # Build and save zero-shot model
    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder.model)
    delattr(image_encoder.model, 'transformer')
    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

    zeroshot_checkpoint = os.path.join(args.save, 'zeroshot'+args.train_dataset+'.pt')
    classifier.save(zeroshot_checkpoint)

    # Standard fine-tuning
    args.load = zeroshot_checkpoint
    args.save = os.path.join(args.save, 'finetuned')

    # Mimic eurosat low-res images, val data aug
    train_data_aug = Compose([
        # Resize(64), # resize to 32/64 for Cifar / Eurosat
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])

    finetuned_checkpoint = finetune_fsl(args, train_data_aug)



if __name__ == '__main__':
    args = parse_arguments()
    classifier_tuning(args)



