import glob
import random

import torch
import time, os
import torch.nn as nn

# from src.clip import clip as clip
import clip.clip as clip
from PIL import Image

def main(img_path, class_num):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_preprocess, val_preprocess = clip.load('RN50', 'cuda', jit=False)

    aircraft_classnames = \
    ['a 707-320', 'a 727-200', 'a 737-200', 'a 737-300', 'a 737-400', 'a 737-500', 'a 737-600', 'a 737-700', 'a 737-800', 'a 737-900', 'a 747-100', 'a 747-200', 'a 747-300', 'a 747-400', 'a 757-200', 'a 757-300', 'a 767-200', 'a 767-300', 'a 767-400', 'a 777-200', 'a 777-300', 'a A300B4', 'a A310', 'a A318', 'a A319', 'a A320', 'a A321', 'a A330-200', 'a A330-300', 'a A340-200', 'a A340-300', 'a A340-500', 'a A340-600', 'a A380', 'a ATR-42', 'a ATR-72', 'a An-12', 'a BAE 146-200', 'a BAE 146-300', 'a BAE-125', 'a Beechcraft 1900', 'a Boeing 717', 'a C-130', 'a C-47', 'a CRJ-200', 'a CRJ-700', 'a CRJ-900', 'a Cessna 172', 'a Cessna 208', 'a Cessna 525', 'a Cessna 560', 'a Challenger 600', 'a DC-10', 'a DC-3', 'a DC-6', 'a DC-8', 'a DC-9-30', 'a DH-82', 'a DHC-1', 'a DHC-6', 'a DHC-8-100', 'a DHC-8-300', 'a DR-400', 'a Dornier 328', 'a E-170', 'a E-190', 'a E-195', 'a EMB-120', 'a ERJ 135', 'a ERJ 145', 'a Embraer Legacy 600', 'a Eurofighter Typhoon', 'a F-16A/B', 'a F/A-18', 'a Falcon 2000', 'a Falcon 900', 'a Fokker 100', 'a Fokker 50', 'a Fokker 70', 'a Global Express', 'a Gulfstream IV', 'a Gulfstream V', 'a Hawk T1', 'a Il-76', 'a L-1011', 'a MD-11', 'a MD-80', 'a MD-87', 'a MD-90', 'a Metroliner', 'a Model B200', 'a PA-28', 'a SR-20', 'a Saab 2000', 'a Saab 340', 'a Spitfire', 'a Tornado', 'a Tu-134', 'a Tu-154', 'a Yak-42']

    eurosat_names = \
    ['a Annual Crop Land', 'a Forest', 'a Herbaceous Vegetation Land', 'a Highway or Road', 'a Industrial Building', 'a Pasture Land', 'a Permanent Crop Land', 'a Residential Building', 'a River', 'a Sea or Lake']

    SUN397_names = \
    ['a abbey', 'a airplane cabin', 'a airport terminal', 'a alley', 'a amphitheater', 'a amusement arcade', 'a amusement park', 'a anechoic chamber', 'a apartment building outdoor', 'a apse indoor', 'a aquarium', 'a aqueduct', 'a arch', 'a archive', 'a arrival gate outdoor', 'a art gallery', 'a art school', 'a art studio', 'a assembly line', 'a athletic field outdoor', 'a atrium public', 'a attic', 'a auditorium', 'a auto factory', 'a badlands', 'a badminton court indoor', 'a baggage claim', 'a bakery shop', 'a balcony exterior', 'a balcony interior', 'a ball pit', 'a ballroom', 'a bamboo forest', 'a banquet hall', 'a bar', 'a barn', 'a barndoor', 'a baseball field', 'a basement', 'a basilica', 'a basketball court outdoor', 'a bathroom', 'a batters box', 'a bayou', 'a bazaar indoor', 'a bazaar outdoor', 'a beach', 'a beauty salon', 'a bedroom', 'a berth', 'a biology laboratory', 'a bistro indoor', 'a boardwalk', 'a boat deck', 'a boathouse', 'a bookstore', 'a booth indoor', 'a botanical garden', 'a bow window indoor', 'a bow window outdoor', 'a bowling alley', 'a boxing ring', 'a brewery indoor', 'a bridge', 'a building facade', 'a bullring', 'a burial chamber', 'a bus interior', 'a butchers shop', 'a butte', 'a cabin outdoor', 'a cafeteria', 'a campsite', 'a campus', 'a canal natural', 'a canal urban', 'a candy store', 'a canyon', 'a car interior backseat', 'a car interior frontseat', 'a carrousel', 'a casino indoor', 'a castle', 'a catacomb', 'a cathedral indoor', 'a cathedral outdoor', 'a cavern indoor', 'a cemetery', 'a chalet', 'a cheese factory', 'a chemistry lab', 'a chicken coop indoor', 'a chicken coop outdoor', 'a childs room', 'a church indoor', 'a church outdoor', 'a classroom', 'a clean room', 'a cliff', 'a cloister indoor', 'a closet', 'a clothing store', 'a coast', 'a cockpit', 'a coffee shop', 'a computer room', 'a conference center', 'a conference room', 'a construction site', 'a control room', 'a control tower outdoor', 'a corn field', 'a corral', 'a corridor', 'a cottage garden', 'a courthouse', 'a courtroom', 'a courtyard', 'a covered bridge exterior', 'a creek', 'a crevasse', 'a crosswalk', 'a cubicle office', 'a dam', 'a delicatessen', 'a dentists office', 'a desert sand', 'a desert vegetation', 'a diner indoor', 'a diner outdoor', 'a dinette home', 'a dinette vehicle', 'a dining car', 'a dining room', 'a discotheque', 'a dock', 'a doorway outdoor', 'a dorm room', 'a driveway', 'a driving range outdoor', 'a drugstore', 'a electrical substation', 'a elevator door', 'a elevator interior', 'a elevator shaft', 'a engine room', 'a escalator indoor', 'a excavation', 'a factory indoor', 'a fairway', 'a fastfood restaurant', 'a field cultivated', 'a field wild', 'a fire escape', 'a fire station', 'a firing range indoor', 'a fishpond', 'a florist shop indoor', 'a food court', 'a forest broadleaf', 'a forest needleleaf', 'a forest path', 'a forest road', 'a formal garden', 'a fountain', 'a galley', 'a game room', 'a garage indoor', 'a garbage dump', 'a gas station', 'a gazebo exterior', 'a general store indoor', 'a general store outdoor', 'a gift shop', 'a golf course', 'a greenhouse indoor', 'a greenhouse outdoor', 'a gymnasium indoor', 'a hangar indoor', 'a hangar outdoor', 'a harbor', 'a hayfield', 'a heliport', 'a herb garden', 'a highway', 'a hill', 'a home office', 'a hospital', 'a hospital room', 'a hot spring', 'a hot tub outdoor', 'a hotel outdoor', 'a hotel room', 'a house', 'a hunting lodge outdoor', 'a ice cream parlor', 'a ice floe', 'a ice shelf', 'a ice skating rink indoor', 'a ice skating rink outdoor', 'a iceberg', 'a igloo', 'a industrial area', 'a inn outdoor', 'a islet', 'a jacuzzi indoor', 'a jail cell', 'a jail indoor', 'a jewelry shop', 'a kasbah', 'a kennel indoor', 'a kennel outdoor', 'a kindergarden classroom', 'a kitchen', 'a kitchenette', 'a labyrinth outdoor', 'a lake natural', 'a landfill', 'a landing deck', 'a laundromat', 'a lecture room', 'a library indoor', 'a library outdoor', 'a lido deck outdoor', 'a lift bridge', 'a lighthouse', 'a limousine interior', 'a living room', 'a lobby', 'a lock chamber', 'a locker room', 'a mansion', 'a manufactured home', 'a market indoor', 'a market outdoor', 'a marsh', 'a martial arts gym', 'a mausoleum', 'a medina', 'a moat water', 'a monastery outdoor', 'a mosque indoor', 'a mosque outdoor', 'a motel', 'a mountain', 'a mountain snowy', 'a movie theater indoor', 'a museum indoor', 'a music store', 'a music studio', 'a nuclear power plant outdoor', 'a nursery', 'a oast house', 'a observatory outdoor', 'a ocean', 'a office', 'a office building', 'a oil refinery outdoor', 'a oilrig', 'a operating room', 'a orchard', 'a outhouse outdoor', 'a pagoda', 'a palace', 'a pantry', 'a park', 'a parking garage indoor', 'a parking garage outdoor', 'a parking lot', 'a parlor', 'a pasture', 'a patio', 'a pavilion', 'a pharmacy', 'a phone booth', 'a physics laboratory', 'a picnic area', 'a pilothouse indoor', 'a planetarium outdoor', 'a playground', 'a playroom', 'a plaza', 'a podium indoor', 'a podium outdoor', 'a pond', 'a poolroom establishment', 'a poolroom home', 'a power plant outdoor', 'a promenade deck', 'a pub indoor', 'a pulpit', 'a putting green', 'a racecourse', 'a raceway', 'a raft', 'a railroad track', 'a rainforest', 'a reception', 'a recreation room', 'a residential neighborhood', 'a restaurant', 'a restaurant kitchen', 'a restaurant patio', 'a rice paddy', 'a riding arena', 'a river', 'a rock arch', 'a rope bridge', 'a ruin', 'a runway', 'a sandbar', 'a sandbox', 'a sauna', 'a schoolhouse', 'a sea cliff', 'a server room', 'a shed', 'a shoe shop', 'a shopfront', 'a shopping mall indoor', 'a shower', 'a skatepark', 'a ski lodge', 'a ski resort', 'a ski slope', 'a sky', 'a skyscraper', 'a slum', 'a snowfield', 'a squash court', 'a stable', 'a stadium baseball', 'a stadium football', 'a stage indoor', 'a staircase', 'a street', 'a subway interior', 'a subway station platform', 'a supermarket', 'a sushi bar', 'a swamp', 'a swimming pool indoor', 'a swimming pool outdoor', 'a synagogue indoor', 'a synagogue outdoor', 'a television studio', 'a temple east asia', 'a temple south asia', 'a tennis court indoor', 'a tennis court outdoor', 'a tent outdoor', 'a theater indoor procenium', 'a theater indoor seats', 'a thriftshop', 'a throne room', 'a ticket booth', 'a toll plaza', 'a topiary garden', 'a tower', 'a toyshop', 'a track outdoor', 'a train railway', 'a train station platform', 'a tree farm', 'a tree house', 'a trench', 'a underwater coral reef', 'a utility room', 'a valley', 'a van interior', 'a vegetable garden', 'a veranda', 'a veterinarians office', 'a viaduct', 'a videostore', 'a village', 'a vineyard', 'a volcano', 'a volleyball court indoor', 'a volleyball court outdoor', 'a waiting room', 'a warehouse indoor', 'a water tower', 'a waterfall block', 'a waterfall fan', 'a waterfall plunge', 'a watering hole', 'a wave', 'a wet bar', 'a wheat field', 'a wind farm', 'a windmill', 'a wine cellar barrel storage', 'a wine cellar bottle storage', 'a wrestling ring indoor', 'a yard', 'a youth hostel']

    caltech256_names = \
    ['a ak47', 'a american-flag', 'a backpack', 'a baseball-bat', 'a baseball-glove', 'a basketball-hoop', 'a bat', 'a bathtub', 'a bear', 'a beer-mug', 'a billiards', 'a binoculars', 'a birdbath', 'a blimp', 'a bonsai-101', 'a boom-box', 'a bowling-ball', 'a bowling-pin', 'a boxing-glove', 'a brain-101', 'a breadmaker', 'a buddha-101', 'a bulldozer', 'a butterfly', 'a cactus', 'a cake', 'a calculator', 'a camel', 'a cannon', 'a canoe', 'a car-tire', 'a cartman', 'a cd', 'a centipede', 'a cereal-box', 'a chandelier-101', 'a chess-board', 'a chimp', 'a chopsticks', 'a cockroach', 'a coffee-mug', 'a coffin', 'a coin', 'a comet', 'a computer-keyboard', 'a computer-monitor', 'a computer-mouse', 'a conch', 'a cormorant', 'a covered-wagon', 'a cowboy-hat', 'a crab-101', 'a desk-globe', 'a diamond-ring', 'a dice', 'a dog', 'a dolphin-101', 'a doorknob', 'a drinking-straw', 'a duck', 'a dumb-bell', 'a eiffel-tower', 'a electric-guitar-101', 'a elephant-101', 'a elk', 'a ewer-101', 'a eyeglasses', 'a fern', 'a fighter-jet', 'a fire-extinguisher', 'a fire-hydrant', 'a fire-truck', 'a fireworks', 'a flashlight', 'a floppy-disk', 'a football-helmet', 'a french-horn', 'a fried-egg', 'a frisbee', 'a frog', 'a frying-pan', 'a galaxy', 'a gas-pump', 'a giraffe', 'a goat', 'a golden-gate-bridge', 'a goldfish', 'a golf-ball', 'a goose', 'a gorilla', 'a grand-piano-101', 'a grapes', 'a grasshopper', 'a guitar-pick', 'a hamburger', 'a hammock', 'a harmonica', 'a harp', 'a harpsichord', 'a hawksbill-101', 'a head-phones', 'a helicopter-101', 'a hibiscus', 'a homer-simpson', 'a horse', 'a horseshoe-crab', 'a hot-air-balloon', 'a hot-dog', 'a hot-tub', 'a hourglass', 'a house-fly', 'a human-skeleton', 'a hummingbird', 'a ibis-101', 'a ice-cream-cone', 'a iguana', 'a ipod', 'a iris', 'a jesus-christ', 'a joy-stick', 'a kangaroo-101', 'a kayak', 'a ketch-101', 'a killer-whale', 'a knife', 'a ladder', 'a laptop-101', 'a lathe', 'a leopards-101', 'a license-plate', 'a lightbulb', 'a light-house', 'a lightning', 'a llama-101', 'a mailbox', 'a mandolin', 'a mars', 'a mattress', 'a megaphone', 'a menorah-101', 'a microscope', 'a microwave', 'a minaret', 'a minotaur', 'a motorbikes-101', 'a mountain-bike', 'a mushroom', 'a mussels', 'a necktie', 'a octopus', 'a ostrich', 'a owl', 'a palm-pilot', 'a palm-tree', 'a paperclip', 'a paper-shredder', 'a pci-card', 'a penguin', 'a people', 'a pez-dispenser', 'a photocopier', 'a picnic-table', 'a playing-card', 'a porcupine', 'a pram', 'a praying-mantis', 'a pyramid', 'a raccoon', 'a radio-telescope', 'a rainbow', 'a refrigerator', 'a revolver-101', 'a rifle', 'a rotary-phone', 'a roulette-wheel', 'a saddle', 'a saturn', 'a school-bus', 'a scorpion-101', 'a screwdriver', 'a segway', 'a self-propelled-lawn-mower', 'a sextant', 'a sheet-music', 'a skateboard', 'a skunk', 'a skyscraper', 'a smokestack', 'a snail', 'a snake', 'a sneaker', 'a snowmobile', 'a soccer-ball', 'a socks', 'a soda-can', 'a spaghetti', 'a speed-boat', 'a spider', 'a spoon', 'a stained-glass', 'a starfish-101', 'a steering-wheel', 'a stirrups', 'a sunflower-101', 'a superman', 'a sushi', 'a swan', 'a swiss-army-knife', 'a sword', 'a syringe', 'a tambourine', 'a teapot', 'a teddy-bear', 'a teepee', 'a telephone-box', 'a tennis-ball', 'a tennis-court', 'a tennis-racket', 'a theodolite', 'a toaster', 'a tomato', 'a tombstone', 'a top-hat', 'a touring-bike', 'a tower-pisa', 'a traffic-light', 'a treadmill', 'a triceratops', 'a tricycle', 'a trilobite-101', 'a tripod', 'a t-shirt', 'a tuning-fork', 'a tweezer', 'a umbrella-101', 'a unicorn', 'a vcr', 'a video-projector', 'a washing-machine', 'a watch-101', 'a waterfall', 'a watermelon', 'a welding-mask', 'a wheelbarrow', 'a windmill', 'a wine-bottle', 'a xylophone', 'a yarmulke', 'a yo-yo', 'a zebra', 'a airplanes-101', 'a car-side-101', 'a faces-easy-101', 'a greyhound', 'a tennis-shoes', 'a toad', 'a clutter']

    cifar10_names = \
    [
        'a airplane',
        'a automobile',
        'a bird',
        'a cat',
        'a deer',
        'a dog',
        'a frog',
        'a horse',
        'a ship',
        'a truck',
    ]

    cifar100_names_old = ['a apple', 'a aquarium fish', 'a baby', 'a bear', 'a beaver', 'a bed', 'a bee', 'a beetle', 'a bicycle', 'a bottles', 'a bowls', 'a boy', 'a bridge', 'a bus', 'a butterfly', 'a camel', 'a cans', 'a castle', 'a caterpillar', 'a cattle', 'a chair', 'a chimpanzee', 'a clock', 'a cloud', 'a cockroach', 'a computer keyboard', 'a couch', 'a crab', 'a crocodile', 'a cups', 'a dinosaur', 'a dolphin', 'a elephant', 'a flatfish', 'a forest', 'a fox', 'a girl', 'a hamster', 'a house', 'a kangaroo', 'a lamp', 'a lawn-mower', 'a leopard', 'a lion', 'a lizard', 'a lobster', 'a man', 'a maple', 'a motorcycle', 'a mountain', 'a mouse', 'a mushrooms', 'a oak', 'a oranges', 'a orchids', 'a otter', 'a palm', 'a pears', 'a pickup truck', 'a pine', 'a plain', 'a plates', 'a poppies', 'a porcupine', 'a possum', 'a rabbit', 'a raccoon', 'a ray', 'a road', 'a rocket', 'a roses', 'a sea', 'a seal', 'a shark', 'a shrew', 'a skunk', 'a skyscraper', 'a snail', 'a snake', 'a spider', 'a squirrel', 'a streetcar', 'a sunflowers', 'a sweet peppers', 'a table', 'a tank', 'a telephone', 'a television', 'a tiger', 'a tractor', 'a train', 'a trout', 'a tulips', 'a turtle', 'a wardrobe', 'a whale', 'a willow', 'a wolf', 'a woman', 'a worm']

    cifar100_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


    caltech101_names = \
    ['a Faces', 'a Faces easy', 'a Leopards', 'a Motorbikes', 'a accordion', 'a airplanes', 'a anchor', 'a ant', 'a barrel', 'a bass', 'a beaver', 'a binocular', 'a bonsai', 'a brain', 'a brontosaurus', 'a buddha', 'a butterfly', 'a camera', 'a cannon', 'a car side', 'a ceiling fan', 'a cellphone', 'a chair', 'a chandelier', 'a cougar body', 'a cougar face', 'a crab', 'a crayfish', 'a crocodile', 'a crocodile head', 'a cup', 'a dalmatian', 'a dollar bill', 'a dolphin', 'a dragonfly', 'a electric guitar', 'a elephant', 'a emu', 'a euphonium', 'a ewer', 'a ferry', 'a flamingo', 'a flamingo head', 'a garfield', 'a gerenuk', 'a gramophone', 'a grand piano', 'a hawksbill', 'a headphone', 'a hedgehog', 'a helicopter', 'a ibis', 'a inline skate', 'a joshua tree', 'a kangaroo', 'a ketch', 'a lamp', 'a laptop', 'a llama', 'a lobster', 'a lotus', 'a mandolin', 'a mayfly', 'a menorah', 'a metronome', 'a minaret', 'a nautilus', 'a octopus', 'a okapi', 'a pagoda', 'a panda', 'a pigeon', 'a pizza', 'a platypus', 'a pyramid', 'a revolver', 'a rhino', 'a rooster', 'a saxophone', 'a schooner', 'a scissors', 'a scorpion', 'a sea horse', 'a snoopy', 'a soccer ball', 'a stapler', 'a starfish', 'a stegosaurus', 'a stop sign', 'a strawberry', 'a sunflower', 'a tick', 'a trilobite', 'a umbrella', 'a watch', 'a water lilly', 'a wheelchair', 'a wild cat', 'a windsor chair', 'a wrench', 'a yin yang']

    cub_names = \
    ['a Black footed Albatross', 'a Laysan Albatross', 'a Sooty Albatross', 'a Groove billed Ani', 'a Crested Auklet', 'a Least Auklet',
     'a Parakeet Auklet', 'a Rhinoceros Auklet', 'a Brewer Blackbird', 'a Red winged Blackbird', 'a Rusty Blackbird', 'a Yellow headed Blackbird',
     'a Bobolink', 'a Indigo Bunting', 'a Lazuli Bunting', 'a Painted Bunting', 'a Cardinal', 'a Spotted Catbird', 'a Gray Catbird',
     'a Yellow breasted Chat', 'a Eastern Towhee', 'a Chuck will Widow', 'a Brandt Cormorant', 'a Red faced Cormorant', 'a Pelagic Cormorant',
     'a Bronzed Cowbird', 'a Shiny Cowbird', 'a Brown Creeper', 'a American Crow', 'a Fish Crow', 'a Black billed Cuckoo', 'a Mangrove Cuckoo',
     'a Yellow billed Cuckoo', 'a Gray crowned Rosy Finch', 'a Purple Finch', 'a Northern Flicker', 'a Acadian Flycatcher', 'a Great Crested Flycatcher',
     'a Least Flycatcher', 'a Olive sided Flycatcher', 'a Scissor tailed Flycatcher', 'a Vermilion Flycatcher', 'a Yellow bellied Flycatcher',
     'a Frigatebird', 'a Northern Fulmar', 'a Gadwall', 'a American Goldfinch', 'a European Goldfinch', 'a Boat tailed Grackle', 'a Eared Grebe',
     'a Horned Grebe', 'a Pied billed Grebe', 'a Western Grebe', 'a Blue Grosbeak', 'a Evening Grosbeak', 'a Pine Grosbeak', 'a Rose breasted Grosbeak',
     'a Pigeon Guillemot', 'a California Gull', 'a Glaucous winged Gull', 'a Heermann Gull', 'a Herring Gull', 'a Ivory Gull', 'a Ring billed Gull',
     'a Slaty backed Gull', 'a Western Gull', 'a Anna Hummingbird', 'a Ruby throated Hummingbird', 'a Rufous Hummingbird',
     'a Green Violetear', 'a Long tailed Jaeger', 'a Pomarine Jaeger', 'a Blue Jay', 'a Florida Jay', 'a Green Jay', 'a Dark eyed Junco',
     'a Tropical Kingbird', 'a Gray Kingbird', 'a Belted Kingfisher', 'a Green Kingfisher', 'a Pied Kingfisher', 'a Ringed Kingfisher',
     'a White breasted Kingfisher', 'a Red legged Kittiwake', 'a Horned Lark', 'a Pacific Loon', 'a Mallard', 'a Western Meadowlark',
     'a Hooded Merganser', 'a Red breasted Merganser', 'a Mockingbird', 'a Nighthawk', 'a Clark Nutcracker', 'a White breasted Nuthatch',
     'a Baltimore Oriole', 'a Hooded Oriole', 'a Orchard Oriole', 'a Scott Oriole', 'a Ovenbird', 'a Brown Pelican', 'a White Pelican',
     'a Western Wood Pewee', 'a Sayornis', 'a American Pipit', 'a Whip poor Will', 'a Horned Puffin', 'a Common Raven', 'a White necked Raven',
     'a American Redstart', 'a Geococcyx', 'a Loggerhead Shrike', 'a Great Grey Shrike', 'a Baird Sparrow', 'a Black throated Sparrow',
     'a Brewer Sparrow', 'a Chipping Sparrow', 'a Clay colored Sparrow', 'a House Sparrow', 'a Field Sparrow', 'a Fox Sparrow',
     'a Grasshopper Sparrow', 'a Harris Sparrow', 'a Henslow Sparrow', 'a Le Conte Sparrow', 'a Lincoln Sparrow', 'a Nelson Sharp tailed Sparrow',
     'a Savannah Sparrow', 'a Seaside Sparrow', 'a Song Sparrow', 'a Tree Sparrow', 'a Vesper Sparrow', 'a White crowned Sparrow',
     'a White throated Sparrow', 'a Cape Glossy Starling', 'a Bank Swallow', 'a Barn Swallow', 'a Cliff Swallow', 'a Tree Swallow',
     'a Scarlet Tanager', 'a Summer Tanager', 'a Artic Tern', 'a Black Tern', 'a Caspian Tern', 'a Common Tern', 'a Elegant Tern',
     'a Forsters Tern', 'a Least Tern', 'a Green tailed Towhee', 'a Brown Thrasher', 'a Sage Thrasher', 'a Black capped Vireo',
     'a Blue headed Vireo', 'a Philadelphia Vireo', 'a Red eyed Vireo', 'a Warbling Vireo', 'a White eyed Vireo',
     'a Yellow throated Vireo', 'a Bay breasted Warbler', 'a Black and white Warbler', 'a Black throated Blue Warbler',
     'a Blue winged Warbler', 'a Canada Warbler', 'a Cape May Warbler', 'a Cerulean Warbler', 'a Chestnut sided Warbler',
     'a Golden winged Warbler', 'a Hooded Warbler', 'a Kentucky Warbler', 'a Magnolia Warbler', 'a Mourning Warbler',
     'a Myrtle Warbler', 'a Nashville Warbler', 'a Orange crowned Warbler', 'a Palm Warbler', 'a Pine Warbler', 'a Prairie Warbler',
     'a Prothonotary Warbler', 'a Swainson Warbler', 'a Tennessee Warbler', 'a Wilson Warbler', 'a Worm eating Warbler',
     'a Yellow Warbler', 'a Northern Waterthrush', 'a Louisiana Waterthrush', 'a Bohemian Waxwing', 'a Cedar Waxwing',
     'a American Three toed Woodpecker', 'a Pileated Woodpecker', 'a Red bellied Woodpecker', 'a Red cockaded Woodpecker',
     'a Red headed Woodpecker', 'a Downy Woodpecker', 'a Bewick Wren', 'a Cactus Wren', 'a Carolina Wren', 'a House Wren',
     'a Marsh Wren', 'a Rock Wren', 'a Winter Wren', 'a Common Yellowthroat']

    openai_classnames = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
            "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
            "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
            "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
            "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
            "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
            "box turtle", "banded gecko", "green iguana", "Carolina anole",
            "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
            "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
            "American alligator", "triceratops", "worm snake", "ring-necked snake",
            "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
            "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
            "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
            "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
            "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
            "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
            "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
            "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
            "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
            "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
            "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
            "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
            "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
            "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
            "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
            "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
            "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
            "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
            "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
            "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
            "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
            "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
            "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
            "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
            "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
            "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
            "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
            "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
            "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
            "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
            "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
            "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
            "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
            "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
            "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
            "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
            "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
            "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
            "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
            "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
            "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
            "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
            "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
            "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
            "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
            "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
            "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
            "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
            "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
            "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
            "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
            "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
            "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
            "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
            "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
            "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
            "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
            "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
            "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
            "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
            "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
            "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
            "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
            "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
            "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
            "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
            "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
            "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
            "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
            "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
            "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
            "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
            "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
            "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
            "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
            "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
            "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
            "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
            "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
            "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
            "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
            "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
            "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
            "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
            "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
            "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
            "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
            "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
            "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
            "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
            "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
            "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
            "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
            "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
            "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
            "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
            "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
            "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
            "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
            "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
            "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
            "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
            "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
            "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
            "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
            "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
            "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
            "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
            "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
            "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
            "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
            "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
            "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
            "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
            "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
            "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
            "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
            "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
            "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
            "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
            "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
            "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
            "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
            "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
            "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
            "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
            "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
            "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
            "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
            "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
            "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
            "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
            "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
            "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
            "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
            "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
            "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
            "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
            "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
            "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
            "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
            "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
            "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
            "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
            "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
            "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
            "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
            "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
            "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
            "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
            "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
            "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
            "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
            "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
            "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
        ]

    # prompts = ["a photo of a " + name + ", a type of aircraft." for name in aircraft_classnames]
    # prompts = ["a photo of " + name + "." for name in SUN397_names]
    # prompts = ["a photo of " + name + "." for name in caltech256_names]
    # prompts = ["a photo of " + name + "." for name in cifar10_names]
    # prompts = ["a photo of a " + name + "." for name in openai_classnames]
    # prompts = ["a photo of " + name + ", a type of bird." for name in cub_names]
    # prompts = ["a photo of a " + name + "." for name in cifar100_names]
    # prompts = ["a photo of " + name + "." for name in caltech101_names]
    prompts = ["a centered satellite photo of " + name + "." for name in eurosat_names]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()


    class TextEncoder(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.token_embedding = clip_model.token_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
            self.dtype = clip_model.dtype

        def forward(self, tokenized_prompts):
            x = self.token_embedding(tokenized_prompts).type(self.dtype)

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

            return x


    class CLIP_selector(nn.Module):
        def __init__(self, clip_model, train_preprocess, val_preprocess, tokenized_prompts):
            super().__init__()
            # self.prompt_learner = PromptLearner(args, classnames, clip_model)
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.image_encoder = clip_model.visual
            self.text_encoder = TextEncoder(clip_model)
            self.logit_scale = clip_model.logit_scale
            self.dtype = clip_model.dtype
            self.train_preprocess = train_preprocess
            self.val_preprocess = val_preprocess
            self.tokenized_prompts = tokenized_prompts

        def forward(self, image):
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

                # prompts = self.prompt_learner()
                tokenized_prompts = self.tokenized_prompts
                text_features = self.text_encoder(tokenized_prompts)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

            return logits

    def select_by_clip_thres_RS(cls_ims, shot, clip_selector, cls_idx, thres=0.2):
        above_thres_paths = []
        dict_results = {}

        for im_path in cls_ims:
            image = val_preprocess(Image.open(im_path)).unsqueeze(0).to(device)
            logits = clip_selector(image)

            confidence = torch.softmax(logits, dim=1)
            confidence_class = confidence[:,cls_idx].item()
            if confidence_class > thres:
                above_thres_paths.append(im_path)
            dict_results[im_path] = confidence_class

        sorted_results = {k: v for k, v in sorted(dict_results.items(), key=lambda item: item[1], reverse=True)}

        if len(above_thres_paths) < shot:
            selected = list(sorted_results.keys())[:shot]
            print('too few above threshold, return highest confidence samples')
        else:
            selected = random.sample(above_thres_paths, k=shot)
            print('enough above threshold, return random sampled samples')

        return selected

    clip_selector = CLIP_selector(model, train_preprocess, val_preprocess, tokenized_prompts)

    train_cls = sorted(glob.glob('{}/train/*'.format(img_path)))
    shot=800
    threshold = 1/class_num

    n_shot_ds = img_path+'_'+str(shot)+'shot'
    os.system('mkdir -p {}/train'.format(n_shot_ds))

    cls_idx = -1
    for cls in train_cls:
        cls_idx += 1
        print(cls_idx)
        cls_ims = sorted(glob.glob(cls + '/*'))
        selected_ims = select_by_clip_thres_RS(cls_ims, shot, clip_selector, cls_idx, thres=threshold)

        os.system('mkdir -p {}/train/{}'.format(n_shot_ds, cls.split('/')[-1]))
        for ims in selected_ims:
            os.system('cp {} {}/train/{}'.format(ims, n_shot_ds, cls.split('/')[-1]))

if __name__ == "__main__":
    import os
    import sys
    img_path = sys.argv[1]
    class_num = sys.argv[2]

    main(img_path, class_num)

