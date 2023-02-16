from src.templates.utils import append_proper_article

aircraft_template = [
    lambda c: f"a photo of a {c}, a type of aircraft.",
    # lambda c: f"a photo of the {c}, a type of aircraft."
]

birds_template = [
    lambda c: f"a photo of a {c}, a type of bird."
]

eurosat_template = [
    lambda c: f"a centered satellite photo of {c}."
]

# eurosat_template = [
#     lambda c: f"a centered satellite photo of {c}.",
#     lambda c: f'a centered satellite photo of a {c}.',
#     lambda c: f'a centered satellite photo of the {c}.',
# ]


flowers_template = [
    lambda c: f"a photo of a {c}, a type of flower."
]

food_template = [
    lambda c: f"a photo of a {c}, a type of food."
]

pets_template = [
    lambda c: f"a photo of a {c}, a type of pet."
]

imagenet_template = [
 lambda c: f"itap of a {c}.",
 lambda c: f"a bad photo of the {c}.",
 lambda c: f"a origami {c}.",
 lambda c: f"a photo of the large {c}.",
 lambda c: f"a {c} in a video game.",
 lambda c: f"art of the {c}.",
 lambda c: f"a photo of the small {c}."]

cifar100_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar10_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

sun_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

dtd_template = [
    lambda c: f"{c} texture."
]

# dtd_template = [
#     lambda c: f'a photo of a {c} texture.',
#     lambda c: f'a photo of a {c} pattern.',
#     lambda c: f'a photo of a {c} thing.',
#     lambda c: f'a photo of a {c} object.',
#     lambda c: f'a photo of the {c} texture.',
#     lambda c: f'a photo of the {c} pattern.',
#     lambda c: f'a photo of the {c} thing.',
#     lambda c: f'a photo of the {c} object.',
# ]