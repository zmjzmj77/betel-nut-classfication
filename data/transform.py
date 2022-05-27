from cvtorchvision import cvtransforms

def transforms():
    transform = cvtransforms.Compose([
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.RandomAffine(30, fillcolor=(255, 255, 255)),
        # cvtransforms.ColorJitter(0.3, 0.3, 0.3, 0.5),
        cvtransforms.RandomVerticalFlip(),
        cvtransforms.ToTensor(),
        cvtransforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform