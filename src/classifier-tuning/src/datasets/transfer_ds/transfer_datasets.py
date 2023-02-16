from robustness.datasets import DataSet, CIFAR, ImageNet
from robustness.tools import folder

import torch
import torch as ch
from . import constants as cs
from torchvision.datasets import CIFAR100

from .caltech import Caltech101, Caltech256
import os

from . import aircraft, food_101, dtd
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import random

from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

name_imagenet_r = ['goldfish', 'great white shark', 'hammerhead shark', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 'bald eagle', 'vulture', 'smooth newt', 'axolotl', 'tree frog', 'green iguana', 'chameleon', 'Indian cobra', 'scorpion', 'tarantula', 'centipede', 'peafowl', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black swan', 'koala', 'jellyfish', 'snail', 'American lobster', 'hermit crab', 'flamingo', 'great egret', 'pelican', 'king penguin', 'grey whale', 'killer whale', 'sea lion', 'Chihuahua', 'Shih Tzu', 'Afghan Hound', 'Basset Hound', 'Beagle', 'Bloodhound', 'Italian Greyhound', 'Whippet', 'Weimaraner', 'Yorkshire Terrier', 'Boston Terrier', 'Scottish Terrier', 'West Highland White Terrier', 'Golden Retriever', 'Labrador Retriever', 'Cocker Spaniel', 'collie', 'Border Collie', 'Rottweiler', 'German Shepherd Dog', 'Boxer', 'French Bulldog', 'St. Bernard', 'Siberian Husky', 'Dalmatian', 'pug', 'Pomeranian', 'Chow Chow', 'Pembroke Welsh Corgi', 'Toy Poodle', 'Standard Poodle', 'grey wolf', 'hyena', 'red fox', 'tabby cat', 'leopard', 'snow leopard', 'lion', 'tiger', 'cheetah', 'polar bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'praying mantis', 'dragonfly', 'monarch butterfly', 'starfish', 'cottontail rabbit', 'porcupine', 'fox squirrel', 'beaver', 'guinea pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'giant panda', 'eel', 'clownfish', 'pufferfish', 'accordion', 'ambulance', 'assault rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer glass', 'binoculars', 'birdhouse', 'bow tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile phone', 'cowboy hat', 'electric guitar', 'fire truck', 'flute', 'gas mask or respirator', 'grand piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab coat', 'lawn mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup truck', 'pirate ship', 'revolver', 'rugby ball', 'sandal', 'saxophone', 'school bus', 'schooner', 'shield', 'soccer ball', 'space shuttle', 'spider web', 'steam locomotive', 'scarf', 'submarine', 'tank', 'tennis ball', 'tractor', 'trombone', 'vase', 'violin', 'military aircraft', 'wine bottle', 'ice cream', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'cabbage', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'Granny Smith apple', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball player', 'scuba diver', 'acorn']


cars_names = ['a FIAT 500 Convertible 2012', 'a Ferrari FF Coupe 2012', 'a Ferrari California Convertible 2012', 'a Ferrari 458 Italia Convertible 2012', 'a Ferrari 458 Italia Coupe 2012', 'a Fisker Karma Sedan 2012', 'a Ford F-450 Super Duty Crew Cab 2012', 'a Ford Mustang Convertible 2007', 'a Ford Freestar Minivan 2007', 'a Ford Expedition EL SUV 2009', 'a Aston Martin Virage Convertible 2012', 'a Ford Edge SUV 2012', 'a Ford Ranger SuperCab 2011', 'a Ford GT Coupe 2006', 'a Ford F-150 Regular Cab 2012', 'a Ford F-150 Regular Cab 2007', 'a Ford Focus Sedan 2007', 'a Ford E-Series Wagon Van 2012', 'a Ford Fiesta Sedan 2012', 'a GMC Terrain SUV 2012', 'a GMC Savana Van 2012', 'a Aston Martin Virage Coupe 2012', 'a GMC Yukon Hybrid SUV 2012', 'a GMC Acadia SUV 2012', 'a GMC Canyon Extended Cab 2012', 'a Geo Metro Convertible 1993', 'a HUMMER H3T Crew Cab 2010', 'a HUMMER H2 SUT Crew Cab 2009', 'a Honda Odyssey Minivan 2012', 'a Honda Odyssey Minivan 2007', 'a Honda Accord Coupe 2012', 'a Honda Accord Sedan 2012', 'a Audi RS 4 Convertible 2008', 'a Hyundai Veloster Hatchback 2012', 'a Hyundai Santa Fe SUV 2012', 'a Hyundai Tucson SUV 2012', 'a Hyundai Veracruz SUV 2012', 'a Hyundai Sonata Hybrid Sedan 2012', 'a Hyundai Elantra Sedan 2007', 'a Hyundai Accent Sedan 2012', 'a Hyundai Genesis Sedan 2012', 'a Hyundai Sonata Sedan 2012', 'a Hyundai Elantra Touring Hatchback 2012', 'a Audi A5 Coupe 2012', 'a Hyundai Azera Sedan 2012', 'a Infiniti G Coupe IPL 2012', 'a Infiniti QX56 SUV 2011', 'a Isuzu Ascender SUV 2008', 'a Jaguar XK XKR 2012', 'a Jeep Patriot SUV 2012', 'a Jeep Wrangler SUV 2012', 'a Jeep Liberty SUV 2012', 'a Jeep Grand Cherokee SUV 2012', 'a Jeep Compass SUV 2012', 'a Audi TTS Coupe 2012', 'a Lamborghini Reventon Coupe 2008', 'a Lamborghini Aventador Coupe 2012', 'a Lamborghini Gallardo LP 570-4 Superleggera 2012', 'a Lamborghini Diablo Coupe 2001', 'a Land Rover Range Rover SUV 2012', 'a Land Rover LR2 SUV 2012', 'a Lincoln Town Car Sedan 2011', 'a MINI Cooper Roadster Convertible 2012', 'a Maybach Landaulet Convertible 2012', 'a Mazda Tribute SUV 2011', 'a Audi R8 Coupe 2012', 'a McLaren MP4-12C Coupe 2012', 'a Mercedes-Benz 300-Class Convertible 1993', 'a Mercedes-Benz C-Class Sedan 2012', 'a Mercedes-Benz SL-Class Coupe 2009', 'a Mercedes-Benz E-Class Sedan 2012', 'a Mercedes-Benz S-Class Sedan 2012', 'a Mercedes-Benz Sprinter Van 2012', 'a Mitsubishi Lancer Sedan 2012', 'a Nissan Leaf Hatchback 2012', 'a Nissan NV Passenger Van 2012', 'a Audi V8 Sedan 1994', 'a Nissan Juke Hatchback 2012', 'a Nissan 240SX Coupe 1998', 'a Plymouth Neon Coupe 1999', 'a Porsche Panamera Sedan 2012', 'a Ram C/V Cargo Van Minivan 2012', 'a Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'a Rolls-Royce Ghost Sedan 2012', 'a Rolls-Royce Phantom Sedan 2012', 'a Scion xD Hatchback 2012', 'a Spyker C8 Convertible 2009', 'a Audi 100 Sedan 1994', 'a Spyker C8 Coupe 2009', 'a Suzuki Aerio Sedan 2007', 'a Suzuki Kizashi Sedan 2012', 'a Suzuki SX4 Hatchback 2012', 'a Suzuki SX4 Sedan 2012', 'a Tesla Model S Sedan 2012', 'a Toyota Sequoia SUV 2012', 'a Toyota Camry Sedan 2012', 'a Toyota Corolla Sedan 2012', 'a Toyota 4Runner SUV 2012', 'a Audi 100 Wagon 1994', 'a Volkswagen Golf Hatchback 2012', 'a Volkswagen Golf Hatchback 1991', 'a Volkswagen Beetle Hatchback 2012', 'a Volvo C30 Hatchback 2012', 'a Volvo 240 Sedan 1993', 'a Volvo XC90 SUV 2007', 'a smart fortwo Convertible 2012', 'a Audi TT Hatchback 2011', 'a AM General Hummer SUV 2000', 'a Audi S6 Sedan 2011', 'a Audi S5 Convertible 2012', 'a Audi S5 Coupe 2012', 'a Audi S4 Sedan 2012', 'a Audi S4 Sedan 2007', 'a Audi TT RS Coupe 2012', 'a BMW ActiveHybrid 5 Sedan 2012', 'a BMW 1 Series Convertible 2012', 'a BMW 1 Series Coupe 2012', 'a BMW 3 Series Sedan 2012', 'a Acura RL Sedan 2012', 'a BMW 3 Series Wagon 2012', 'a BMW 6 Series Convertible 2007', 'a BMW X5 SUV 2007', 'a BMW X6 SUV 2012', 'a BMW M3 Coupe 2012', 'a BMW M5 Sedan 2010', 'a BMW M6 Convertible 2010', 'a BMW X3 SUV 2012', 'a BMW Z4 Convertible 2012', 'a Bentley Continental Supersports Conv. Convertible 2012', 'a Acura TL Sedan 2012', 'a Bentley Arnage Sedan 2009', 'a Bentley Mulsanne Sedan 2011', 'a Bentley Continental GT Coupe 2012', 'a Bentley Continental GT Coupe 2007', 'a Bentley Continental Flying Spur Sedan 2007', 'a Bugatti Veyron 16.4 Convertible 2009', 'a Bugatti Veyron 16.4 Coupe 2009', 'a Buick Regal GS 2012', 'a Buick Rainier SUV 2007', 'a Buick Verano Sedan 2012', 'a Acura TL Type-S 2008', 'a Buick Enclave SUV 2012', 'a Cadillac CTS-V Sedan 2012', 'a Cadillac SRX SUV 2012', 'a Cadillac Escalade EXT Crew Cab 2007', 'a Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'a Chevrolet Corvette Convertible 2012', 'a Chevrolet Corvette ZR1 2012', 'a Chevrolet Corvette Ron Fellows Edition Z06 2007', 'a Chevrolet Traverse SUV 2012', 'a Chevrolet Camaro Convertible 2012', 'a Acura TSX Sedan 2012', 'a Chevrolet HHR SS 2010', 'a Chevrolet Impala Sedan 2007', 'a Chevrolet Tahoe Hybrid SUV 2012', 'a Chevrolet Sonic Sedan 2012', 'a Chevrolet Express Cargo Van 2007', 'a Chevrolet Avalanche Crew Cab 2012', 'a Chevrolet Cobalt SS 2010', 'a Chevrolet Malibu Hybrid Sedan 2010', 'a Chevrolet TrailBlazer SS 2009', 'a Chevrolet Silverado 2500HD Regular Cab 2012', 'a Acura Integra Type R 2001', 'a Chevrolet Silverado 1500 Classic Extended Cab 2007', 'a Chevrolet Express Van 2007', 'a Chevrolet Monte Carlo Coupe 2007', 'a Chevrolet Malibu Sedan 2007', 'a Chevrolet Silverado 1500 Extended Cab 2012', 'a Chevrolet Silverado 1500 Regular Cab 2012', 'a Chrysler Aspen SUV 2009', 'a Chrysler Sebring Convertible 2010', 'a Chrysler Town and Country Minivan 2012', 'a Chrysler 300 SRT-8 2010', 'a Acura ZDX Hatchback 2012', 'a Chrysler Crossfire Convertible 2008', 'a Chrysler PT Cruiser Convertible 2008', 'a Daewoo Nubira Wagon 2002', 'a Dodge Caliber Wagon 2012', 'a Dodge Caliber Wagon 2007', 'a Dodge Caravan Minivan 1997', 'a Dodge Ram Pickup 3500 Crew Cab 2010', 'a Dodge Ram Pickup 3500 Quad Cab 2009', 'a Dodge Sprinter Cargo Van 2009', 'a Dodge Journey SUV 2012', 'a Aston Martin V8 Vantage Convertible 2012', 'a Dodge Dakota Crew Cab 2010', 'a Dodge Dakota Club Cab 2007', 'a Dodge Magnum Wagon 2008', 'a Dodge Challenger SRT8 2011', 'a Dodge Durango SUV 2012', 'a Dodge Durango SUV 2007', 'a Dodge Charger Sedan 2012', 'a Dodge Charger SRT-8 2009', 'a Eagle Talon Hatchback 1998', 'a FIAT 500 Abarth 2012', 'a Aston Martin V8 Vantage Coupe 2012']
cars_names_coop = \
['2000 AM General Hummer SUV', '2012 Acura RL Sedan', '2012 Aston Martin Virage Coupe', '2012 Ferrari FF Coupe', '2012 Ferrari California Convertible', '2012 Ferrari 458 Italia Convertible', '2012 Ferrari 458 Italia Coupe', '2012 Fisker Karma Sedan', '2012 Ford F-450 Super Duty Crew Cab', '2007 Ford Mustang Convertible', '2007 Ford Freestar Minivan', '2009 Ford Expedition EL SUV', '2012 Ford Edge SUV', '2008 Audi RS 4 Convertible', '2011 Ford Ranger SuperCab', '2006 Ford GT Coupe', '2012 Ford F-150 Regular Cab', '2007 Ford F-150 Regular Cab', '2007 Ford Focus Sedan', '2012 Ford E-Series Wagon Van', '2012 Ford Fiesta Sedan', '2012 GMC Terrain SUV', '2012 GMC Savana Van', '2012 GMC Yukon Hybrid SUV', '2012 Audi A5 Coupe', '2012 GMC Acadia SUV', '2012 GMC Canyon Extended Cab', '1993 Geo Metro Convertible', '2010 HUMMER H3T Crew Cab', '2009 HUMMER H2 SUT Crew Cab', '2012 Honda Odyssey Minivan', '2007 Honda Odyssey Minivan', '2012 Honda Accord Coupe', '2012 Honda Accord Sedan', '2012 Hyundai Veloster Hatchback', '2012 Audi TTS Coupe', '2012 Hyundai Santa Fe SUV', '2012 Hyundai Tucson SUV', '2012 Hyundai Veracruz SUV', '2012 Hyundai Sonata Hybrid Sedan', '2007 Hyundai Elantra Sedan', '2012 Hyundai Accent Sedan', '2012 Hyundai Genesis Sedan', '2012 Hyundai Sonata Sedan', '2012 Hyundai Elantra Touring Hatchback', '2012 Hyundai Azera Sedan', '2012 Audi R8 Coupe', '2012 Infiniti G Coupe IPL', '2011 Infiniti QX56 SUV', '2008 Isuzu Ascender SUV', '2012 Jaguar XK XKR', '2012 Jeep Patriot SUV', '2012 Jeep Wrangler SUV', '2012 Jeep Liberty SUV', '2012 Jeep Grand Cherokee SUV', '2012 Jeep Compass SUV', '2008 Lamborghini Reventon Coupe', '1994 Audi V8 Sedan', '2012 Lamborghini Aventador Coupe', '2012 Lamborghini Gallardo LP 570-4 Superleggera', '2001 Lamborghini Diablo Coupe', '2012 Land Rover Range Rover SUV', '2012 Land Rover LR2 SUV', '2011 Lincoln Town Car Sedan', '2012 MINI Cooper Roadster Convertible', '2012 Maybach Landaulet Convertible', '2011 Mazda Tribute SUV', '2012 McLaren MP4-12C Coupe', '1994 Audi 100 Sedan', '1993 Mercedes-Benz 300-Class Convertible', '2012 Mercedes-Benz C-Class Sedan', '2009 Mercedes-Benz SL-Class Coupe', '2012 Mercedes-Benz E-Class Sedan', '2012 Mercedes-Benz S-Class Sedan', '2012 Mercedes-Benz Sprinter Van', '2012 Mitsubishi Lancer Sedan', '2012 Nissan Leaf Hatchback', '2012 Nissan NV Passenger Van', '2012 Nissan Juke Hatchback', '1994 Audi 100 Wagon', '1998 Nissan 240SX Coupe', '1999 Plymouth Neon Coupe', '2012 Porsche Panamera Sedan', '2012 Ram C/V Cargo Van Minivan', '2012 Rolls-Royce Phantom Drophead Coupe Convertible', '2012 Rolls-Royce Ghost Sedan', '2012 Rolls-Royce Phantom Sedan', '2012 Scion xD Hatchback', '2009 Spyker C8 Convertible', '2009 Spyker C8 Coupe', '2011 Audi TT Hatchback', '2007 Suzuki Aerio Sedan', '2012 Suzuki Kizashi Sedan', '2012 Suzuki SX4 Hatchback', '2012 Suzuki SX4 Sedan', '2012 Tesla Model S Sedan', '2012 Toyota Sequoia SUV', '2012 Toyota Camry Sedan', '2012 Toyota Corolla Sedan', '2012 Toyota 4Runner SUV', '2012 Volkswagen Golf Hatchback', '2011 Audi S6 Sedan', '1991 Volkswagen Golf Hatchback', '2012 Volkswagen Beetle Hatchback', '2012 Volvo C30 Hatchback', '1993 Volvo 240 Sedan', '2007 Volvo XC90 SUV', '2012 smart fortwo Convertible', '2012 Acura TL Sedan', '2012 Audi S5 Convertible', '2012 Audi S5 Coupe', '2012 Audi S4 Sedan', '2007 Audi S4 Sedan', '2012 Audi TT RS Coupe', '2012 BMW ActiveHybrid 5 Sedan', '2012 BMW 1 Series Convertible', '2012 BMW 1 Series Coupe', '2012 BMW 3 Series Sedan', '2012 BMW 3 Series Wagon', '2008 Acura TL Type-S', '2007 BMW 6 Series Convertible', '2007 BMW X5 SUV', '2012 BMW X6 SUV', '2012 BMW M3 Coupe', '2010 BMW M5 Sedan', '2010 BMW M6 Convertible', '2012 BMW X3 SUV', '2012 BMW Z4 Convertible', '2012 Bentley Continental Supersports Conv. Convertible', '2009 Bentley Arnage Sedan', '2012 Acura TSX Sedan', '2011 Bentley Mulsanne Sedan', '2012 Bentley Continental GT Coupe', '2007 Bentley Continental GT Coupe', '2007 Bentley Continental Flying Spur Sedan', '2009 Bugatti Veyron 16.4 Convertible', '2009 Bugatti Veyron 16.4 Coupe', '2012 Buick Regal GS', '2007 Buick Rainier SUV', '2012 Buick Verano Sedan', '2012 Buick Enclave SUV', '2001 Acura Integra Type R', '2012 Cadillac CTS-V Sedan', '2012 Cadillac SRX SUV', '2007 Cadillac Escalade EXT Crew Cab', '2012 Chevrolet Silverado 1500 Hybrid Crew Cab', '2012 Chevrolet Corvette Convertible', '2012 Chevrolet Corvette ZR1', '2007 Chevrolet Corvette Ron Fellows Edition Z06', '2012 Chevrolet Traverse SUV', '2012 Chevrolet Camaro Convertible', '2010 Chevrolet HHR SS', '2012 Acura ZDX Hatchback', '2007 Chevrolet Impala Sedan', '2012 Chevrolet Tahoe Hybrid SUV', '2012 Chevrolet Sonic Sedan', '2007 Chevrolet Express Cargo Van', '2012 Chevrolet Avalanche Crew Cab', '2010 Chevrolet Cobalt SS', '2010 Chevrolet Malibu Hybrid Sedan', '2009 Chevrolet TrailBlazer SS', '2012 Chevrolet Silverado 2500HD Regular Cab', '2007 Chevrolet Silverado 1500 Classic Extended Cab', '2012 Aston Martin V8 Vantage Convertible', '2007 Chevrolet Express Van', '2007 Chevrolet Monte Carlo Coupe', '2007 Chevrolet Malibu Sedan', '2012 Chevrolet Silverado 1500 Extended Cab', '2012 Chevrolet Silverado 1500 Regular Cab', '2009 Chrysler Aspen SUV', '2010 Chrysler Sebring Convertible', '2012 Chrysler Town and Country Minivan', '2010 Chrysler 300 SRT-8', '2008 Chrysler Crossfire Convertible', '2012 Aston Martin V8 Vantage Coupe', '2008 Chrysler PT Cruiser Convertible', '2002 Daewoo Nubira Wagon', '2012 Dodge Caliber Wagon', '2007 Dodge Caliber Wagon', '1997 Dodge Caravan Minivan', '2010 Dodge Ram Pickup 3500 Crew Cab', '2009 Dodge Ram Pickup 3500 Quad Cab', '2009 Dodge Sprinter Cargo Van', '2012 Dodge Journey SUV', '2010 Dodge Dakota Crew Cab', '2012 Aston Martin Virage Convertible', '2007 Dodge Dakota Club Cab', '2008 Dodge Magnum Wagon', '2011 Dodge Challenger SRT8', '2012 Dodge Durango SUV', '2007 Dodge Durango SUV', '2012 Dodge Charger Sedan', '2009 Dodge Charger SRT-8', '1998 Eagle Talon Hatchback', '2012 FIAT 500 Abarth', '2012 FIAT 500 Convertible']

aircraft_names = \
['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE-125', 'BAE 146-200', 'BAE 146-300', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

birds_names = \
[
    'Acadian Flycatcher',
    'Acorn Woodpecker',
    'Alder Flycatcher',
    'Allens Hummingbird',
    'Altamira Oriole',
    'American Avocet',
    'American Bittern',
    'American Black Duck',
    'American Coot',
    'American Crow',
    'American Dipper',
    'American Golden Plover',
    'American Goldfinch',
    'American Kestrel',
    'American Oystercatcher',
    'American Pipit',
    'American Redstart',
    'American Robin',
    'American Three toed Woodpecker',
    'American Tree Sparrow',
    'American White Pelican',
    'American Wigeon',
    'American Woodcock',
    'Anhinga',
    'Annas Hummingbird',
    'Arctic Tern',
    'Ash throated Flycatcher',
    'Audubons Oriole',
    'Bairds Sandpiper',
    'Bald Eagle',
    'Baltimore Oriole',
    'Band tailed Pigeon',
    'Barn Swallow',
    'Barred Owl',
    'Barrows Goldeneye',
    'Bay breasted Warbler',
    'Bells Vireo',
    'Belted Kingfisher',
    'Bewicks Wren',
    'Black Guillemot',
    'Black Oystercatcher',
    'Black Phoebe',
    'Black Rosy Finch',
    'Black Scoter',
    'Black Skimmer',
    'Black Tern',
    'Black Turnstone',
    'Black Vulture',
    'Black and white Warbler',
    'Black backed Woodpecker',
    'Black bellied Plover',
    'Black billed Cuckoo',
    'Black billed Magpie',
    'Black capped Chickadee',
    'Black chinned Hummingbird',
    'Black chinned Sparrow',
    'Black crested Titmouse',
    'Black crowned Night Heron',
    'Black headed Grosbeak',
    'Black legged Kittiwake',
    'Black necked Stilt',
    'Black throated Blue Warbler',
    'Black throated Gray Warbler',
    'Black throated Green Warbler',
    'Black throated Sparrow',
    'Blackburnian Warbler',
    'Blackpoll Warbler',
    'Blue Grosbeak',
    'Blue Jay',
    'Blue gray Gnatcatcher',
    'Blue headed Vireo',
    'Blue winged Teal',
    'Blue winged Warbler',
    'Boat tailed Grackle',
    'Bobolink',
    'Bohemian Waxwing',
    'Bonapartes Gull',
    'Boreal Chickadee',
    'Brandts Cormorant',
    'Brant',
    'Brewers Blackbird',
    'Brewers Sparrow',
    'Bridled Titmouse',
    'Broad billed Hummingbird',
    'Broad tailed Hummingbird',
    'Broad winged Hawk',
    'Bronzed Cowbird',
    'Brown Creeper',
    'Brown Pelican',
    'Brown Thrasher',
    'Brown capped Rosy Finch',
    'Brown crested Flycatcher',
    'Brown headed Cowbird',
    'Brown headed Nuthatch',
    'Bufflehead',
    'Bullocks Oriole',
    'Burrowing Owl',
    'Bushtit',
    'Cackling Goose',
    'Cactus Wren',
    'California Gull',
    'California Quail',
    'California Thrasher',
    'California Towhee',
    'Calliope Hummingbird',
    'Canada Goose',
    'Canada Warbler',
    'Canvasback',
    'Canyon Towhee',
    'Canyon Wren',
    'Cape May Warbler',
    'Carolina Chickadee',
    'Carolina Wren',
    'Caspian Tern',
    'Cassins Finch',
    'Cassins Kingbird',
    'Cassins Sparrow',
    'Cassins Vireo',
    'Cattle Egret',
    'Cave Swallow',
    'Cedar Waxwing',
    'Cerulean Warbler',
    'Chestnut backed Chickadee',
    'Chestnut collared Longspur',
    'Chestnut sided Warbler',
    'Chihuahuan Raven',
    'Chimney Swift',
    'Chipping Sparrow',
    'Cinnamon Teal',
    'Clapper Rail',
    'Clarks Grebe',
    'Clarks Nutcracker',
    'Clay colored Sparrow',
    'Cliff Swallow',
    'Common Black Hawk',
    'Common Eider',
    'Common Gallinule',
    'Common Goldeneye',
    'Common Grackle',
    'Common Ground Dove',
    'Common Loon',
    'Common Merganser',
    'Common Murre',
    'Common Nighthawk',
    'Common Raven',
    'Common Redpoll',
    'Common Tern',
    'Common Yellowthroat',
    'Connecticut Warbler',
    'Coopers Hawk',
    'Cordilleran Flycatcher',
    'Costas Hummingbird',
    'Couchs Kingbird',
    'Crested Caracara',
    'Curve billed Thrasher',
    'Dark eyed Junco',
    'Dickcissel',
    'Double crested Cormorant',
    'Downy Woodpecker',
    'Dunlin',
    'Dusky Flycatcher',
    'Dusky Grouse',
    'Eared Grebe',
    'Eastern Bluebird',
    'Eastern Kingbird',
    'Eastern Meadowlark',
    'Eastern Phoebe',
    'Eastern Screech Owl',
    'Eastern Towhee',
    'Eastern Wood Pewee',
    'Elegant Trogon',
    'Elf Owl',
    'Eurasian Collared Dove',
    'Eurasian Wigeon',
    'European Starling',
    'Evening Grosbeak',
    'Ferruginous Hawk',
    'Ferruginous Pygmy Owl',
    'Field Sparrow',
    'Fish Crow',
    'Florida Scrub Jay',
    'Forsters Tern',
    'Fox Sparrow',
    'Franklins Gull',
    'Fulvous Whistling Duck',
    'Gadwall',
    'Gambels Quail',
    'Gila Woodpecker',
    'Glaucous Gull',
    'Glaucous winged Gull',
    'Glossy Ibis',
    'Golden Eagle',
    'Golden crowned Kinglet',
    'Golden crowned Sparrow',
    'Golden fronted Woodpecker',
    'Golden winged Warbler',
    'Grasshopper Sparrow',
    'Gray Catbird',
    'Gray Flycatcher',
    'Gray Jay',
    'Gray Kingbird',
    'Gray cheeked Thrush',
    'Gray crowned Rosy Finch',
    'Great Black backed Gull',
    'Great Blue Heron',
    'Great Cormorant',
    'Great Crested Flycatcher',
    'Great Egret',
    'Great Gray Owl',
    'Great Horned Owl',
    'Great Kiskadee',
    'Great tailed Grackle',
    'Greater Prairie Chicken',
    'Greater Roadrunner',
    'Greater Sage Grouse',
    'Greater Scaup',
    'Greater White fronted Goose',
    'Greater Yellowlegs',
    'Green Jay',
    'Green tailed Towhee',
    'Green winged Teal',
    'Groove billed Ani',
    'Gull billed Tern',
    'Hairy Woodpecker',
    'Hammonds Flycatcher',
    'Harlequin Duck',
    'Harriss Hawk',
    'Harriss Sparrow',
    'Heermanns Gull',
    'Henslows Sparrow',
    'Hepatic Tanager',
    'Hermit Thrush',
    'Herring Gull',
    'Hoary Redpoll',
    'Hooded Merganser',
    'Hooded Oriole',
    'Hooded Warbler',
    'Horned Grebe',
    'Horned Lark',
    'House Finch',
    'House Sparrow',
    'House Wren',
    'Huttons Vireo',
    'Iceland Gull',
    'Inca Dove',
    'Indigo Bunting',
    'Killdeer',
    'King Rail',
    'Ladder backed Woodpecker',
    'Lapland Longspur',
    'Lark Bunting',
    'Lark Sparrow',
    'Laughing Gull',
    'Lazuli Bunting',
    'Le Contes Sparrow',
    'Least Bittern',
    'Least Flycatcher',
    'Least Grebe',
    'Least Sandpiper',
    'Least Tern',
    'Lesser Goldfinch',
    'Lesser Nighthawk',
    'Lesser Scaup',
    'Lesser Yellowlegs',
    'Lewiss Woodpecker',
    'Limpkin',
    'Lincolns Sparrow',
    'Little Blue Heron',
    'Loggerhead Shrike',
    'Long billed Curlew',
    'Long billed Dowitcher',
    'Long billed Thrasher',
    'Long eared Owl',
    'Long tailed Duck',
    'Louisiana Waterthrush',
    'Magnificent Frigatebird',
    'Magnolia Warbler',
    'Mallard',
    'Marbled Godwit',
    'Marsh Wren',
    'Merlin',
    'Mew Gull',
    'Mexican Jay',
    'Mississippi Kite',
    'Monk Parakeet',
    'Mottled Duck',
    'Mountain Bluebird',
    'Mountain Chickadee',
    'Mountain Plover',
    'Mourning Dove',
    'Mourning Warbler',
    'Muscovy Duck',
    'Mute Swan',
    'Nashville Warbler',
    'Nelsons Sparrow',
    'Neotropic Cormorant',
    'Northern Bobwhite',
    'Northern Cardinal',
    'Northern Flicker',
    'Northern Gannet',
    'Northern Goshawk',
    'Northern Harrier',
    'Northern Hawk Owl',
    'Northern Mockingbird',
    'Northern Parula',
    'Northern Pintail',
    'Northern Rough winged Swallow',
    'Northern Saw whet Owl',
    'Northern Shrike',
    'Northern Waterthrush',
    'Nuttalls Woodpecker',
    'Oak Titmouse',
    'Olive Sparrow',
    'Olive sided Flycatcher',
    'Orange crowned Warbler',
    'Orchard Oriole',
    'Osprey',
    'Ovenbird',
    'Pacific Golden Plover',
    'Pacific Loon',
    'Pacific Wren',
    'Pacific slope Flycatcher',
    'Painted Bunting',
    'Painted Redstart',
    'Palm Warbler',
    'Pectoral Sandpiper',
    'Peregrine Falcon',
    'Phainopepla',
    'Philadelphia Vireo',
    'Pied billed Grebe',
    'Pigeon Guillemot',
    'Pileated Woodpecker',
    'Pine Grosbeak',
    'Pine Siskin',
    'Pine Warbler',
    'Piping Plover',
    'Plumbeous Vireo',
    'Prairie Falcon',
    'Prairie Warbler',
    'Prothonotary Warbler',
    'Purple Finch',
    'Purple Gallinule',
    'Purple Martin',
    'Purple Sandpiper',
    'Pygmy Nuthatch',
    'Pyrrhuloxia',
    'Red Crossbill',
    'Red Knot',
    'Red Phalarope',
    'Red bellied Woodpecker',
    'Red breasted Merganser',
    'Red breasted Nuthatch',
    'Red breasted Sapsucker',
    'Red cockaded Woodpecker',
    'Red eyed Vireo',
    'Red headed Woodpecker',
    'Red naped Sapsucker',
    'Red necked Grebe',
    'Red necked Phalarope',
    'Red shouldered Hawk',
    'Red tailed Hawk',
    'Red throated Loon',
    'Red winged Blackbird',
    'Reddish Egret',
    'Redhead',
    'Ring billed Gull',
    'Ring necked Duck',
    'Ring necked Pheasant',
    'Rock Pigeon',
    'Rock Ptarmigan',
    'Rock Sandpiper',
    'Rock Wren',
    'Rose breasted Grosbeak',
    'Roseate Tern',
    'Rosss Goose',
    'Rough legged Hawk',
    'Royal Tern',
    'Ruby crowned Kinglet',
    'Ruby throated Hummingbird',
    'Ruddy Duck',
    'Ruddy Turnstone',
    'Ruffed Grouse',
    'Rufous Hummingbird',
    'Rufous crowned Sparrow',
    'Rusty Blackbird',
    'Sage Thrasher',
    'Saltmarsh Sparrow',
    'Sanderling',
    'Sandhill Crane',
    'Sandwich Tern',
    'Says Phoebe',
    'Scaled Quail',
    'Scarlet Tanager',
    'Scissor tailed Flycatcher',
    'Scotts Oriole',
    'Seaside Sparrow',
    'Sedge Wren',
    'Semipalmated Plover',
    'Semipalmated Sandpiper',
    'Sharp shinned Hawk',
    'Sharp tailed Grouse',
    'Short billed Dowitcher',
    'Short eared Owl',
    'Snail Kite',
    'Snow Bunting',
    'Snow Goose',
    'Snowy Egret',
    'Snowy Owl',
    'Snowy Plover',
    'Solitary Sandpiper',
    'Song Sparrow',
    'Sooty Grouse',
    'Sora',
    'Spotted Owl',
    'Spotted Sandpiper',
    'Spotted Towhee',
    'Spruce Grouse',
    'Stellers Jay',
    'Stilt Sandpiper',
    'Summer Tanager',
    'Surf Scoter',
    'Surfbird',
    'Swainsons Hawk',
    'Swainsons Thrush',
    'Swallow tailed Kite',
    'Swamp Sparrow',
    'Tennessee Warbler',
    'Thayers Gull',
    'Townsends Solitaire',
    'Townsends Warbler',
    'Tree Swallow',
    'Tricolored Heron',
    'Tropical Kingbird',
    'Trumpeter Swan',
    'Tufted Titmouse',
    'Tundra Swan',
    'Turkey Vulture',
    'Upland Sandpiper',
    'Varied Thrush',
    'Veery',
    'Verdin',
    'Vermilion Flycatcher',
    'Vesper Sparrow',
    'Violet green Swallow',
    'Virginia Rail',
    'Wandering Tattler',
    'Warbling Vireo',
    'Western Bluebird',
    'Western Grebe',
    'Western Gull',
    'Western Kingbird',
    'Western Meadowlark',
    'Western Sandpiper',
    'Western Screech Owl',
    'Western Scrub Jay',
    'Western Tanager',
    'Western Wood Pewee',
    'Whimbrel',
    'White Ibis',
    'White breasted Nuthatch',
    'White crowned Sparrow',
    'White eyed Vireo',
    'White faced Ibis',
    'White headed Woodpecker',
    'White rumped Sandpiper',
    'White tailed Hawk',
    'White tailed Kite',
    'White tailed Ptarmigan',
    'White throated Sparrow',
    'White throated Swift',
    'White winged Crossbill',
    'White winged Dove',
    'White winged Scoter',
    'Wild Turkey',
    'Willet',
    'Williamsons Sapsucker',
    'Willow Flycatcher',
    'Willow Ptarmigan',
    'Wilsons Phalarope',
    'Wilsons Plover',
    'Wilsons Snipe',
    'Wilsons Warbler',
    'Winter Wren',
    'Wood Stork',
    'Wood Thrush',
    'Worm eating Warbler',
    'Wrentit',
    'Yellow Warbler',
    'Yellow bellied Flycatcher',
    'Yellow bellied Sapsucker',
    'Yellow billed Cuckoo',
    'Yellow billed Magpie',
    'Yellow breasted Chat',
    'Yellow crowned Night Heron',
    'Yellow eyed Junco',
    'Yellow headed Blackbird',
    'Yellow rumped Warbler',
    'Yellow throated Vireo',
    'Yellow throated Warbler',
    'Zone tailed Hawk',
]
# ['a Acadian Flycatcher', 'a Acorn Woodpecker', 'a Alder Flycatcher', 'a Allens Hummingbird', 'a Altamira Oriole', 'a American Avocet', 'a American Bittern', 'a American Black Duck', 'a American Coot', 'a American Crow', 'a American Dipper', 'a American Golden Plover', 'a American Goldfinch', 'a American Kestrel', 'a American Oystercatcher', 'a American Pipit', 'a American Redstart', 'a American Robin', 'a American Three toed Woodpecker', 'a American Tree Sparrow', 'a American White Pelican', 'a American Wigeon', 'a American Woodcock', 'a Anhinga', 'a Annas Hummingbird', 'a Arctic Tern', 'a Ash throated Flycatcher', 'a Audubons Oriole', 'a Bairds Sandpiper', 'a Bald Eagle', 'a Baltimore Oriole', 'a Band tailed Pigeon', 'a Barn Swallow', 'a Barred Owl', 'a Barrows Goldeneye', 'a Bay breasted Warbler', 'a Bells Vireo', 'a Belted Kingfisher', 'a Bewicks Wren', 'a Black Guillemot', 'a Black Oystercatcher', 'a Black Phoebe', 'a Black Rosy Finch', 'a Black Scoter', 'a Black Skimmer', 'a Black Tern', 'a Black Turnstone', 'a Black Vulture', 'a Black and white Warbler', 'a Black backed Woodpecker', 'a Black bellied Plover', 'a Black billed Cuckoo', 'a Black billed Magpie', 'a Black capped Chickadee', 'a Black chinned Hummingbird', 'a Black chinned Sparrow', 'a Black crested Titmouse', 'a Black crowned Night Heron', 'a Black headed Grosbeak', 'a Black legged Kittiwake', 'a Black necked Stilt', 'a Black throated Blue Warbler', 'a Black throated Gray Warbler', 'a Black throated Green Warbler', 'a Black throated Sparrow', 'a Blackburnian Warbler', 'a Blackpoll Warbler', 'a Blue Grosbeak', 'a Blue Jay', 'a Blue gray Gnatcatcher', 'a Blue headed Vireo', 'a Blue winged Teal', 'a Blue winged Warbler', 'a Boat tailed Grackle', 'a Bobolink', 'a Bohemian Waxwing', 'a Bonapartes Gull', 'a Boreal Chickadee', 'a Brandts Cormorant', 'a Brant', 'a Brewers Blackbird', 'a Brewers Sparrow', 'a Bridled Titmouse', 'a Broad billed Hummingbird', 'a Broad tailed Hummingbird', 'a Broad winged Hawk', 'a Bronzed Cowbird', 'a Brown Creeper', 'a Brown Pelican', 'a Brown Thrasher', 'a Brown capped Rosy Finch', 'a Brown crested Flycatcher', 'a Brown headed Cowbird', 'a Brown headed Nuthatch', 'a Bufflehead', 'a Bullocks Oriole', 'a Burrowing Owl', 'a Bushtit', 'a Cackling Goose', 'a Cactus Wren', 'a California Gull', 'a California Quail', 'a California Thrasher', 'a California Towhee', 'a Calliope Hummingbird', 'a Canada Goose', 'a Canada Warbler', 'a Canvasback', 'a Canyon Towhee', 'a Canyon Wren', 'a Cape May Warbler', 'a Carolina Chickadee', 'a Carolina Wren', 'a Caspian Tern', 'a Cassins Finch', 'a Cassins Kingbird', 'a Cassins Sparrow', 'a Cassins Vireo', 'a Cattle Egret', 'a Cave Swallow', 'a Cedar Waxwing', 'a Cerulean Warbler', 'a Chestnut backed Chickadee', 'a Chestnut collared Longspur', 'a Chestnut sided Warbler', 'a Chihuahuan Raven', 'a Chimney Swift', 'a Chipping Sparrow', 'a Cinnamon Teal', 'a Clapper Rail', 'a Clarks Grebe', 'a Clarks Nutcracker', 'a Clay colored Sparrow', 'a Cliff Swallow', 'a Common Black Hawk', 'a Common Eider', 'a Common Gallinule', 'a Common Goldeneye', 'a Common Grackle', 'a Common Ground Dove', 'a Common Loon', 'a Common Merganser', 'a Common Murre', 'a Common Nighthawk', 'a Common Raven', 'a Common Redpoll', 'a Common Tern', 'a Common Yellowthroat', 'a Connecticut Warbler', 'a Coopers Hawk', 'a Cordilleran Flycatcher', 'a Costas Hummingbird', 'a Couchs Kingbird', 'a Crested Caracara', 'a Curve billed Thrasher', 'a Dark eyed Junco', 'a Dickcissel', 'a Double crested Cormorant', 'a Downy Woodpecker', 'a Dunlin', 'a Dusky Flycatcher', 'a Dusky Grouse', 'a Eared Grebe', 'a Eastern Bluebird', 'a Eastern Kingbird', 'a Eastern Meadowlark', 'a Eastern Phoebe', 'a Eastern Screech Owl', 'a Eastern Towhee', 'a Eastern Wood Pewee', 'a Elegant Trogon', 'a Elf Owl', 'a Eurasian Collared Dove', 'a Eurasian Wigeon', 'a European Starling', 'a Evening Grosbeak', 'a Ferruginous Hawk', 'a Ferruginous Pygmy Owl', 'a Field Sparrow', 'a Fish Crow', 'a Florida Scrub Jay', 'a Forsters Tern', 'a Fox Sparrow', 'a Franklins Gull', 'a Fulvous Whistling Duck', 'a Gadwall', 'a Gambels Quail', 'a Gila Woodpecker', 'a Glaucous Gull', 'a Glaucous winged Gull', 'a Glossy Ibis', 'a Golden Eagle', 'a Golden crowned Kinglet', 'a Golden crowned Sparrow', 'a Golden fronted Woodpecker', 'a Golden winged Warbler', 'a Grasshopper Sparrow', 'a Gray Catbird', 'a Gray Flycatcher', 'a Gray Jay', 'a Gray Kingbird', 'a Gray cheeked Thrush', 'a Gray crowned Rosy Finch', 'a Great Black backed Gull', 'a Great Blue Heron', 'a Great Cormorant', 'a Great Crested Flycatcher', 'a Great Egret', 'a Great Gray Owl', 'a Great Horned Owl', 'a Great Kiskadee', 'a Great tailed Grackle', 'a Greater Prairie Chicken', 'a Greater Roadrunner', 'a Greater Sage Grouse', 'a Greater Scaup', 'a Greater White fronted Goose', 'a Greater Yellowlegs', 'a Green Jay', 'a Green tailed Towhee', 'a Green winged Teal', 'a Groove billed Ani', 'a Gull billed Tern', 'a Hairy Woodpecker', 'a Hammonds Flycatcher', 'a Harlequin Duck', 'a Harriss Hawk', 'a Harriss Sparrow', 'a Heermanns Gull', 'a Henslows Sparrow', 'a Hepatic Tanager', 'a Hermit Thrush', 'a Herring Gull', 'a Hoary Redpoll', 'a Hooded Merganser', 'a Hooded Oriole', 'a Hooded Warbler', 'a Horned Grebe', 'a Horned Lark', 'a House Finch', 'a House Sparrow', 'a House Wren', 'a Huttons Vireo', 'a Iceland Gull', 'a Inca Dove', 'a Indigo Bunting', 'a Killdeer', 'a King Rail', 'a Ladder backed Woodpecker', 'a Lapland Longspur', 'a Lark Bunting', 'a Lark Sparrow', 'a Laughing Gull', 'a Lazuli Bunting', 'a Le Contes Sparrow', 'a Least Bittern', 'a Least Flycatcher', 'a Least Grebe', 'a Least Sandpiper', 'a Least Tern', 'a Lesser Goldfinch', 'a Lesser Nighthawk', 'a Lesser Scaup', 'a Lesser Yellowlegs', 'a Lewiss Woodpecker', 'a Limpkin', 'a Lincolns Sparrow', 'a Little Blue Heron', 'a Loggerhead Shrike', 'a Long billed Curlew', 'a Long billed Dowitcher', 'a Long billed Thrasher', 'a Long eared Owl', 'a Long tailed Duck', 'a Louisiana Waterthrush', 'a Magnificent Frigatebird', 'a Magnolia Warbler', 'a Mallard', 'a Marbled Godwit', 'a Marsh Wren', 'a Merlin', 'a Mew Gull', 'a Mexican Jay', 'a Mississippi Kite', 'a Monk Parakeet', 'a Mottled Duck', 'a Mountain Bluebird', 'a Mountain Chickadee', 'a Mountain Plover', 'a Mourning Dove', 'a Mourning Warbler', 'a Muscovy Duck', 'a Mute Swan', 'a Nashville Warbler', 'a Nelsons Sparrow', 'a Neotropic Cormorant', 'a Northern Bobwhite', 'a Northern Cardinal', 'a Northern Flicker', 'a Northern Gannet', 'a Northern Goshawk', 'a Northern Harrier', 'a Northern Hawk Owl', 'a Northern Mockingbird', 'a Northern Parula', 'a Northern Pintail', 'a Northern Rough winged Swallow', 'a Northern Saw whet Owl', 'a Northern Shrike', 'a Northern Waterthrush', 'a Nuttalls Woodpecker', 'a Oak Titmouse', 'a Olive Sparrow', 'a Olive sided Flycatcher', 'a Orange crowned Warbler', 'a Orchard Oriole', 'a Osprey', 'a Ovenbird', 'a Pacific Golden Plover', 'a Pacific Loon', 'a Pacific Wren', 'a Pacific slope Flycatcher', 'a Painted Bunting', 'a Painted Redstart', 'a Palm Warbler', 'a Pectoral Sandpiper', 'a Peregrine Falcon', 'a Phainopepla', 'a Philadelphia Vireo', 'a Pied billed Grebe', 'a Pigeon Guillemot', 'a Pileated Woodpecker', 'a Pine Grosbeak', 'a Pine Siskin', 'a Pine Warbler', 'a Piping Plover', 'a Plumbeous Vireo', 'a Prairie Falcon', 'a Prairie Warbler', 'a Prothonotary Warbler', 'a Purple Finch', 'a Purple Gallinule', 'a Purple Martin', 'a Purple Sandpiper', 'a Pygmy Nuthatch', 'a Pyrrhuloxia', 'a Red Crossbill', 'a Red Knot', 'a Red Phalarope', 'a Red bellied Woodpecker', 'a Red breasted Merganser', 'a Red breasted Nuthatch', 'a Red breasted Sapsucker', 'a Red cockaded Woodpecker', 'a Red eyed Vireo', 'a Red headed Woodpecker', 'a Red naped Sapsucker', 'a Red necked Grebe', 'a Red necked Phalarope', 'a Red shouldered Hawk', 'a Red tailed Hawk', 'a Red throated Loon', 'a Red winged Blackbird', 'a Reddish Egret', 'a Redhead', 'a Ring billed Gull', 'a Ring necked Duck', 'a Ring necked Pheasant', 'a Rock Pigeon', 'a Rock Ptarmigan', 'a Rock Sandpiper', 'a Rock Wren', 'a Rose breasted Grosbeak', 'a Roseate Tern', 'a Rosss Goose', 'a Rough legged Hawk', 'a Royal Tern', 'a Ruby crowned Kinglet', 'a Ruby throated Hummingbird', 'a Ruddy Duck', 'a Ruddy Turnstone', 'a Ruffed Grouse', 'a Rufous Hummingbird', 'a Rufous crowned Sparrow', 'a Rusty Blackbird', 'a Sage Thrasher', 'a Saltmarsh Sparrow', 'a Sanderling', 'a Sandhill Crane', 'a Sandwich Tern', 'a Says Phoebe', 'a Scaled Quail', 'a Scarlet Tanager', 'a Scissor tailed Flycatcher', 'a Scotts Oriole', 'a Seaside Sparrow', 'a Sedge Wren', 'a Semipalmated Plover', 'a Semipalmated Sandpiper', 'a Sharp shinned Hawk', 'a Sharp tailed Grouse', 'a Short billed Dowitcher', 'a Short eared Owl', 'a Snail Kite', 'a Snow Bunting', 'a Snow Goose', 'a Snowy Egret', 'a Snowy Owl', 'a Snowy Plover', 'a Solitary Sandpiper', 'a Song Sparrow', 'a Sooty Grouse', 'a Sora', 'a Spotted Owl', 'a Spotted Sandpiper', 'a Spotted Towhee', 'a Spruce Grouse', 'a Stellers Jay', 'a Stilt Sandpiper', 'a Summer Tanager', 'a Surf Scoter', 'a Surfbird', 'a Swainsons Hawk', 'a Swainsons Thrush', 'a Swallow tailed Kite', 'a Swamp Sparrow', 'a Tennessee Warbler', 'a Thayers Gull', 'a Townsends Solitaire', 'a Townsends Warbler', 'a Tree Swallow', 'a Tricolored Heron', 'a Tropical Kingbird', 'a Trumpeter Swan', 'a Tufted Titmouse', 'a Tundra Swan', 'a Turkey Vulture', 'a Upland Sandpiper', 'a Varied Thrush', 'a Veery', 'a Verdin', 'a Vermilion Flycatcher', 'a Vesper Sparrow', 'a Violet green Swallow', 'a Virginia Rail', 'a Wandering Tattler', 'a Warbling Vireo', 'a Western Bluebird', 'a Western Grebe', 'a Western Gull', 'a Western Kingbird', 'a Western Meadowlark', 'a Western Sandpiper', 'a Western Screech Owl', 'a Western Scrub Jay', 'a Western Tanager', 'a Western Wood Pewee', 'a Whimbrel', 'a White Ibis', 'a White breasted Nuthatch', 'a White crowned Sparrow', 'a White eyed Vireo', 'a White faced Ibis', 'a White headed Woodpecker', 'a White rumped Sandpiper', 'a White tailed Hawk', 'a White tailed Kite', 'a White tailed Ptarmigan', 'a White throated Sparrow', 'a White throated Swift', 'a White winged Crossbill', 'a White winged Dove', 'a White winged Scoter', 'a Wild Turkey', 'a Willet', 'a Williamsons Sapsucker', 'a Willow Flycatcher', 'a Willow Ptarmigan', 'a Wilsons Phalarope', 'a Wilsons Plover', 'a Wilsons Snipe', 'a Wilsons Warbler', 'a Winter Wren', 'a Wood Stork', 'a Wood Thrush', 'a Worm eating Warbler', 'a Wrentit', 'a Yellow Warbler', 'a Yellow bellied Flycatcher', 'a Yellow bellied Sapsucker', 'a Yellow billed Cuckoo', 'a Yellow billed Magpie', 'a Yellow breasted Chat', 'a Yellow crowned Night Heron', 'a Yellow eyed Junco', 'a Yellow headed Blackbird', 'a Yellow rumped Warbler', 'a Yellow throated Vireo', 'a Yellow throated Warbler', 'a Zone tailed Hawk']
cifar10_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
caltech101_names = \
['a Faces', 'a Faces easy', 'a Leopards', 'a Motorbikes', 'a accordion', 'a airplanes', 'a anchor', 'a ant', 'a barrel', 'a bass', 'a beaver', 'a binocular', 'a bonsai', 'a brain', 'a brontosaurus', 'a buddha', 'a butterfly', 'a camera', 'a cannon', 'a car side', 'a ceiling fan', 'a cellphone', 'a chair', 'a chandelier', 'a cougar body', 'a cougar face', 'a crab', 'a crayfish', 'a crocodile', 'a crocodile head', 'a cup', 'a dalmatian', 'a dollar bill', 'a dolphin', 'a dragonfly', 'a electric guitar', 'a elephant', 'a emu', 'a euphonium', 'a ewer', 'a ferry', 'a flamingo', 'a flamingo head', 'a garfield', 'a gerenuk', 'a gramophone', 'a grand piano', 'a hawksbill', 'a headphone', 'a hedgehog', 'a helicopter', 'a ibis', 'a inline skate', 'a joshua tree', 'a kangaroo', 'a ketch', 'a lamp', 'a laptop', 'a llama', 'a lobster', 'a lotus', 'a mandolin', 'a mayfly', 'a menorah', 'a metronome', 'a minaret', 'a nautilus', 'a octopus', 'a okapi', 'a pagoda', 'a panda', 'a pigeon', 'a pizza', 'a platypus', 'a pyramid', 'a revolver', 'a rhino', 'a rooster', 'a saxophone', 'a schooner', 'a scissors', 'a scorpion', 'a sea horse', 'a snoopy', 'a soccer ball', 'a stapler', 'a starfish', 'a stegosaurus', 'a stop sign', 'a strawberry', 'a sunflower', 'a tick', 'a trilobite', 'a umbrella', 'a watch', 'a water lilly', 'a wheelchair', 'a wild cat', 'a windsor chair', 'a wrench', 'a yin yang']
caltech101_coop_names = \
['a face', 'a leopard', 'a motorbike', 'a accordion', 'a airplane', 'a anchor', 'a ant', 'a barrel', 'a bass', 'a beaver', 'a binocular', 'a bonsai', 'a brain', 'a brontosaurus', 'a buddha', 'a butterfly', 'a camera', 'a cannon', 'a car side', 'a ceiling fan', 'a cellphone', 'a chair', 'a chandelier', 'a cougar body', 'a cougar face', 'a crab', 'a crayfish', 'a crocodile', 'a crocodile head', 'a cup', 'a dalmatian', 'a dollar bill', 'a dolphin', 'a dragonfly', 'a electric guitar', 'a elephant', 'a emu', 'a euphonium', 'a ewer', 'a ferry', 'a flamingo', 'a flamingo head', 'a garfield', 'a gerenuk', 'a gramophone', 'a grand piano', 'a hawksbill', 'a headphone', 'a hedgehog', 'a helicopter', 'a ibis', 'a inline skate', 'a joshua tree', 'a kangaroo', 'a ketch', 'a lamp', 'a laptop', 'a llama', 'a lobster', 'a lotus', 'a mandolin', 'a mayfly', 'a menorah', 'a metronome', 'a minaret', 'a nautilus', 'a octopus', 'a okapi', 'a pagoda', 'a panda', 'a pigeon', 'a pizza', 'a platypus', 'a pyramid', 'a revolver', 'a rhino', 'a rooster', 'a saxophone', 'a schooner', 'a scissors', 'a scorpion', 'a sea horse', 'a snoopy', 'a soccer ball', 'a stapler', 'a starfish', 'a stegosaurus', 'a stop sign', 'a strawberry', 'a sunflower', 'a tick', 'a trilobite', 'a umbrella', 'a watch', 'a water lilly', 'a wheelchair', 'a wild cat', 'a windsor chair', 'a wrench', 'a yin yang']
caltech101_clip_names = \
    [
    'off-center face',
    'centered face',
    'leopard',
    'motorbike',
    'accordion',
    'airplane',
    'anchor',
    'ant',
    'background',
    'barrel',
    'bass',
    'beaver',
    'binocular',
    'bonsai',
    'brain',
    'brontosaurus',
    'buddha',
    'butterfly',
    'camera',
    'cannon',
    'side of a car',
    'ceiling fan',
    'cellphone',
    'chair',
    'chandelier',
    'body of a cougar cat',
    'face of a cougar cat',
    'crab',
    'crayfish',
    'crocodile',
    'head of a  crocodile',
    'cup',
    'dalmatian',
    'dollar bill',
    'dolphin',
    'dragonfly',
    'electric guitar',
    'elephant',
    'emu',
    'euphonium',
    'ewer',
    'ferry',
    'flamingo',
    'head of a flamingo',
    'garfield',
    'gerenuk',
    'gramophone',
    'grand piano',
    'hawksbill',
    'headphone',
    'hedgehog',
    'helicopter',
    'ibis',
    'inline skate',
    'joshua tree',
    'kangaroo',
    'ketch',
    'lamp',
    'laptop',
    'llama',
    'lobster',
    'lotus',
    'mandolin',
    'mayfly',
    'menorah',
    'metronome',
    'minaret',
    'nautilus',
    'octopus',
    'okapi',
    'pagoda',
    'panda',
    'pigeon',
    'pizza',
    'platypus',
    'pyramid',
    'revolver',
    'rhino',
    'rooster',
    'saxophone',
    'schooner',
    'scissors',
    'scorpion',
    'sea horse',
    'snoopy (cartoon beagle)',
    'soccer ball',
    'stapler',
    'starfish',
    'stegosaurus',
    'stop sign',
    'strawberry',
    'sunflower',
    'tick',
    'trilobite',
    'umbrella',
    'watch',
    'water lilly',
    'wheelchair',
    'wild cat',
    'windsor chair',
    'wrench',
    'yin and yang symbol',
]
caltech256_names = \
['a ak47', 'a american-flag', 'a backpack', 'a baseball-bat', 'a baseball-glove', 'a basketball-hoop', 'a bat', 'a bathtub', 'a bear', 'a beer-mug', 'a billiards', 'a binoculars', 'a birdbath', 'a blimp', 'a bonsai-101', 'a boom-box', 'a bowling-ball', 'a bowling-pin', 'a boxing-glove', 'a brain-101', 'a breadmaker', 'a buddha-101', 'a bulldozer', 'a butterfly', 'a cactus', 'a cake', 'a calculator', 'a camel', 'a cannon', 'a canoe', 'a car-tire', 'a cartman', 'a cd', 'a centipede', 'a cereal-box', 'a chandelier-101', 'a chess-board', 'a chimp', 'a chopsticks', 'a cockroach', 'a coffee-mug', 'a coffin', 'a coin', 'a comet', 'a computer-keyboard', 'a computer-monitor', 'a computer-mouse', 'a conch', 'a cormorant', 'a covered-wagon', 'a cowboy-hat', 'a crab-101', 'a desk-globe', 'a diamond-ring', 'a dice', 'a dog', 'a dolphin-101', 'a doorknob', 'a drinking-straw', 'a duck', 'a dumb-bell', 'a eiffel-tower', 'a electric-guitar-101', 'a elephant-101', 'a elk', 'a ewer-101', 'a eyeglasses', 'a fern', 'a fighter-jet', 'a fire-extinguisher', 'a fire-hydrant', 'a fire-truck', 'a fireworks', 'a flashlight', 'a floppy-disk', 'a football-helmet', 'a french-horn', 'a fried-egg', 'a frisbee', 'a frog', 'a frying-pan', 'a galaxy', 'a gas-pump', 'a giraffe', 'a goat', 'a golden-gate-bridge', 'a goldfish', 'a golf-ball', 'a goose', 'a gorilla', 'a grand-piano-101', 'a grapes', 'a grasshopper', 'a guitar-pick', 'a hamburger', 'a hammock', 'a harmonica', 'a harp', 'a harpsichord', 'a hawksbill-101', 'a head-phones', 'a helicopter-101', 'a hibiscus', 'a homer-simpson', 'a horse', 'a horseshoe-crab', 'a hot-air-balloon', 'a hot-dog', 'a hot-tub', 'a hourglass', 'a house-fly', 'a human-skeleton', 'a hummingbird', 'a ibis-101', 'a ice-cream-cone', 'a iguana', 'a ipod', 'a iris', 'a jesus-christ', 'a joy-stick', 'a kangaroo-101', 'a kayak', 'a ketch-101', 'a killer-whale', 'a knife', 'a ladder', 'a laptop-101', 'a lathe', 'a leopards-101', 'a license-plate', 'a lightbulb', 'a light-house', 'a lightning', 'a llama-101', 'a mailbox', 'a mandolin', 'a mars', 'a mattress', 'a megaphone', 'a menorah-101', 'a microscope', 'a microwave', 'a minaret', 'a minotaur', 'a motorbikes-101', 'a mountain-bike', 'a mushroom', 'a mussels', 'a necktie', 'a octopus', 'a ostrich', 'a owl', 'a palm-pilot', 'a palm-tree', 'a paperclip', 'a paper-shredder', 'a pci-card', 'a penguin', 'a people', 'a pez-dispenser', 'a photocopier', 'a picnic-table', 'a playing-card', 'a porcupine', 'a pram', 'a praying-mantis', 'a pyramid', 'a raccoon', 'a radio-telescope', 'a rainbow', 'a refrigerator', 'a revolver-101', 'a rifle', 'a rotary-phone', 'a roulette-wheel', 'a saddle', 'a saturn', 'a school-bus', 'a scorpion-101', 'a screwdriver', 'a segway', 'a self-propelled-lawn-mower', 'a sextant', 'a sheet-music', 'a skateboard', 'a skunk', 'a skyscraper', 'a smokestack', 'a snail', 'a snake', 'a sneaker', 'a snowmobile', 'a soccer-ball', 'a socks', 'a soda-can', 'a spaghetti', 'a speed-boat', 'a spider', 'a spoon', 'a stained-glass', 'a starfish-101', 'a steering-wheel', 'a stirrups', 'a sunflower-101', 'a superman', 'a sushi', 'a swan', 'a swiss-army-knife', 'a sword', 'a syringe', 'a tambourine', 'a teapot', 'a teddy-bear', 'a teepee', 'a telephone-box', 'a tennis-ball', 'a tennis-court', 'a tennis-racket', 'a theodolite', 'a toaster', 'a tomato', 'a tombstone', 'a top-hat', 'a touring-bike', 'a tower-pisa', 'a traffic-light', 'a treadmill', 'a triceratops', 'a tricycle', 'a trilobite-101', 'a tripod', 'a t-shirt', 'a tuning-fork', 'a tweezer', 'a umbrella-101', 'a unicorn', 'a vcr', 'a video-projector', 'a washing-machine', 'a watch-101', 'a waterfall', 'a watermelon', 'a welding-mask', 'a wheelbarrow', 'a windmill', 'a wine-bottle', 'a xylophone', 'a yarmulke', 'a yo-yo', 'a zebra', 'a airplanes-101', 'a car-side-101', 'a faces-easy-101', 'a greyhound', 'a tennis-shoes', 'a toad', 'a clutter']
cifar100_names = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]

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
dtd_names = \
["a banded","a blotchy","a braided","a bubbly","a bumpy","a chequered","a cobwebbed","a cracked","a crosshatched","a crystalline",
 "a dotted","a fibrous","a flecked","a freckled","a frilly","a gauzy","a grid","a grooved","a honeycombed","a interlaced","a knitted",
 "a lacelike","a lined","a marbled","a matted","a meshed","a paisley","a perforated","a pitted","a pleated","a polka-dotted","a porous",
 "a potholed","a scaly","a smeared","a spiralled","a sprinkled","a stained","a stratified","a striped","a studded","a swirly","a veined",
 "a waffled","a woven","a wrinkled","a zigzagged"]

eurosat_names = \
['a Annual Crop Land', 'a Forest', 'a Herbaceous Vegetation Land', 'a Highway or Road', 'a Industrial Building', 'a Pasture Land', 'a Permanent Crop Land', 'a Residential Building', 'a River', 'a Sea or Lake']

eurosat_names_coop = \
[
    'annual crop land',
    'forest',
    'brushland or shrubland',
    'highway or road',
    'industrial buildings or commercial buildings',
    'pasture land',
    'permanent crop land',
    'residential buildings or homes or apartments',
    'river',
    'lake or sea',
]
# ['a Annual Crop Land', 'a Forest', 'a Herbaceous Vegetation Land', 'a Highway or Road', 'a Industrial Buildings', 'a Pasture Land', 'a Permanent Crop Land', 'a Residential Buildings', 'a River', 'a Sea or Lake']

flowers_names = \
['a pink primrose', 'a globe thistle', 'a blanket flower', 'a trumpet creeper', 'a blackberry lily', 'a snapdragon', "a colt's foot", 'a king protea', 'a spear thistle', 'a yellow iris', 'a globe-flower', 'a purple coneflower', 'a peruvian lily', 'a balloon flower', 'a hard-leaved pocket orchid', 'a giant white arum lily', 'a fire lily', 'a pincushion flower', 'a fritillary', 'a red ginger', 'a grape hyacinth', 'a corn poppy', 'a prince of wales feathers', 'a stemless gentian', 'a artichoke', 'a canterbury bells', 'a sweet william', 'a carnation', 'a garden phlox', 'a love in the mist', 'a mexican aster', 'a alpine sea holly', 'a ruby-lipped cattleya', 'a cape flower', 'a great masterwort', 'a siam tulip', 'a sweet pea', 'a lenten rose', 'a barbeton daisy', 'a daffodil', 'a sword lily', 'a poinsettia', 'a bolero deep blue', 'a wallflower', 'a marigold', 'a buttercup', 'a oxeye daisy', 'a english marigold', 'a common dandelion', 'a petunia', 'a wild pansy', 'a primula', 'a sunflower', 'a pelargonium', 'a bishop of llandaff', 'a gaura', 'a geranium', 'a orange dahlia', 'a tiger lily', 'a pink-yellow dahlia', 'a cautleya spicata', 'a japanese anemone', 'a black-eyed susan', 'a silverbush', 'a californian poppy', 'a osteospermum', 'a spring crocus', 'a bearded iris', 'a windflower', 'a moon orchid', 'a tree poppy', 'a gazania', 'a azalea', 'a water lily', 'a rose', 'a thorn apple', 'a morning glory', 'a passion flower', 'a lotus', 'a toad lily', 'a bird of paradise', 'a anthurium', 'a frangipani', 'a clematis', 'a hibiscus', 'a columbine', 'a desert-rose', 'a tree mallow', 'a magnolia', 'a cyclamen', 'a watercress', 'a monkshood', 'a canna lily', 'a hippeastrum', 'a bee balm', 'a ball moss', 'a foxglove', 'a bougainvillea', 'a camellia', 'a mallow', 'a mexican petunia', 'a bromelia']

food_names = \
[
    'apple pie',
    'baby back ribs',
    'baklava',
    'beef carpaccio',
    'beef tartare',
    'beet salad',
    'beignets',
    'bibimbap',
    'bread pudding',
    'breakfast burrito',
    'bruschetta',
    'caesar salad',
    'cannoli',
    'caprese salad',
    'carrot cake',
    'ceviche',
    'cheese plate',
    'cheesecake',
    'chicken curry',
    'chicken quesadilla',
    'chicken wings',
    'chocolate cake',
    'chocolate mousse',
    'churros',
    'clam chowder',
    'club sandwich',
    'crab cakes',
    'creme brulee',
    'croque madame',
    'cup cakes',
    'deviled eggs',
    'donuts',
    'dumplings',
    'edamame',
    'eggs benedict',
    'escargots',
    'falafel',
    'filet mignon',
    'fish and chips',
    'foie gras',
    'french fries',
    'french onion soup',
    'french toast',
    'fried calamari',
    'fried rice',
    'frozen yogurt',
    'garlic bread',
    'gnocchi',
    'greek salad',
    'grilled cheese sandwich',
    'grilled salmon',
    'guacamole',
    'gyoza',
    'hamburger',
    'hot and sour soup',
    'hot dog',
    'huevos rancheros',
    'hummus',
    'ice cream',
    'lasagna',
    'lobster bisque',
    'lobster roll sandwich',
    'macaroni and cheese',
    'macarons',
    'miso soup',
    'mussels',
    'nachos',
    'omelette',
    'onion rings',
    'oysters',
    'pad thai',
    'paella',
    'pancakes',
    'panna cotta',
    'peking duck',
    'pho',
    'pizza',
    'pork chop',
    'poutine',
    'prime rib',
    'pulled pork sandwich',
    'ramen',
    'ravioli',
    'red velvet cake',
    'risotto',
    'samosa',
    'sashimi',
    'scallops',
    'seaweed salad',
    'shrimp and grits',
    'spaghetti bolognese',
    'spaghetti carbonara',
    'spring rolls',
    'steak',
    'strawberry shortcake',
    'sushi',
    'tacos',
    'takoyaki',
    'tiramisu',
    'tuna tartare',
    'waffles',
] # clip
# ['a apple pie', 'a baby back ribs', 'a baklava', 'a beef carpaccio', 'a beef tartare', 'a beet salad', 'a beignets', 'a bibimbap', 'a bread pudding', 'a breakfast burrito', 'a bruschetta', 'a caesar salad', 'a cannoli', 'a caprese salad', 'a carrot cake', 'a ceviche', 'a cheese plate', 'a cheesecake', 'a chicken curry', 'a chicken quesadilla', 'a chicken wings', 'a chocolate cake', 'a chocolate mousse', 'a churros', 'a clam chowder', 'a club sandwich', 'a crab cakes', 'a creme brulee', 'a croque madame', 'a cup cakes', 'a deviled eggs', 'a donuts', 'a dumplings', 'a edamame', 'a eggs benedict', 'a escargots', 'a falafel', 'a filet mignon', 'a fish and chips', 'a foie gras', 'a french fries', 'a french onion soup', 'a french toast', 'a fried calamari', 'a fried rice', 'a frozen yogurt', 'a garlic bread', 'a gnocchi', 'a greek salad', 'a grilled cheese sandwich', 'a grilled salmon', 'a guacamole', 'a gyoza', 'a hamburger', 'a hot and sour soup', 'a hot dog', 'a huevos rancheros', 'a hummus', 'a ice cream', 'a lasagna', 'a lobster bisque', 'a lobster roll sandwich', 'a macaroni and cheese', 'a macarons', 'a miso soup', 'a mussels', 'a nachos', 'a omelette', 'a onion rings', 'a oysters', 'a pad thai', 'a paella', 'a pancakes', 'a panna cotta', 'a peking duck', 'a pho', 'a pizza', 'a pork chop', 'a poutine', 'a prime rib', 'a pulled pork sandwich', 'a ramen', 'a ravioli', 'a red velvet cake', 'a risotto', 'a samosa', 'a sashimi', 'a scallops', 'a seaweed salad', 'a shrimp and grits', 'a spaghetti bolognese', 'a spaghetti carbonara', 'a spring rolls', 'a steak', 'a strawberry shortcake', 'a sushi', 'a tacos', 'a takoyaki', 'a tiramisu', 'a tuna tartare', 'a waffles']

pets_names = \
['abyssinian', 'american bulldog', 'chihuahua', 'egyptian mau', 'english cocker spaniel', 'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'american pit bull terrier', 'maine coon', 'miniature pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian blue', 'saint bernard', 'samoyed', 'basset hound', 'scottish terrier', 'shiba inu', 'siamese', 'sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british shorthair']

# ['abyssinian', 'american_bulldog', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'american_pit_bull_terrier', 'maine_coon', 'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue', 'saint_bernard', 'samoyed', 'basset_hound', 'scottish_terrier', 'shiba_inu', 'siamese', 'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair']
# ['a Abyssinian', 'a British Shorthair', 'a chihuahua', 'a Egyptian Mau', 'a english cocker spaniel', 'a english setter', 'a german shorthaired', 'a great pyrenees', 'a havanese', 'a japanese chin', 'a keeshond', 'a american bulldog', 'a leonberger', 'a Maine Coon', 'a miniature pinscher', 'a newfoundland', 'a Persian', 'a pomeranian', 'a pug', 'a Ragdoll', 'a Russian Blue', 'a saint bernard', 'a american pit bull terrier', 'a samoyed', 'a scottish terrier', 'a shiba inu', 'a Siamese', 'a Sphynx', 'a staffordshire bull terrier', 'a wheaten terrier', 'a yorkshire terrier', 'a basset hound', 'a beagle', 'a Bengal', 'a Birman', 'a Bombay', 'a boxer']

SUN397_names = \
['a abbey', 'a airplane cabin', 'a airport terminal', 'a alley', 'a amphitheater', 'a amusement arcade', 'a amusement park', 'a anechoic chamber', 'a apartment building outdoor', 'a apse indoor', 'a aquarium', 'a aqueduct', 'a arch', 'a archive', 'a arrival gate outdoor', 'a art gallery', 'a art school', 'a art studio', 'a assembly line', 'a athletic field outdoor', 'a atrium public', 'a attic', 'a auditorium', 'a auto factory', 'a badlands', 'a badminton court indoor', 'a baggage claim', 'a bakery shop', 'a balcony exterior', 'a balcony interior', 'a ball pit', 'a ballroom', 'a bamboo forest', 'a banquet hall', 'a bar', 'a barn', 'a barndoor', 'a baseball field', 'a basement', 'a basilica', 'a basketball court outdoor', 'a bathroom', 'a batters box', 'a bayou', 'a bazaar indoor', 'a bazaar outdoor', 'a beach', 'a beauty salon', 'a bedroom', 'a berth', 'a biology laboratory', 'a bistro indoor', 'a boardwalk', 'a boat deck', 'a boathouse', 'a bookstore', 'a booth indoor', 'a botanical garden', 'a bow window indoor', 'a bow window outdoor', 'a bowling alley', 'a boxing ring', 'a brewery indoor', 'a bridge', 'a building facade', 'a bullring', 'a burial chamber', 'a bus interior', 'a butchers shop', 'a butte', 'a cabin outdoor', 'a cafeteria', 'a campsite', 'a campus', 'a canal natural', 'a canal urban', 'a candy store', 'a canyon', 'a car interior backseat', 'a car interior frontseat', 'a carrousel', 'a casino indoor', 'a castle', 'a catacomb', 'a cathedral indoor', 'a cathedral outdoor', 'a cavern indoor', 'a cemetery', 'a chalet', 'a cheese factory', 'a chemistry lab', 'a chicken coop indoor', 'a chicken coop outdoor', 'a childs room', 'a church indoor', 'a church outdoor', 'a classroom', 'a clean room', 'a cliff', 'a cloister indoor', 'a closet', 'a clothing store', 'a coast', 'a cockpit', 'a coffee shop', 'a computer room', 'a conference center', 'a conference room', 'a construction site', 'a control room', 'a control tower outdoor', 'a corn field', 'a corral', 'a corridor', 'a cottage garden', 'a courthouse', 'a courtroom', 'a courtyard', 'a covered bridge exterior', 'a creek', 'a crevasse', 'a crosswalk', 'a cubicle office', 'a dam', 'a delicatessen', 'a dentists office', 'a desert sand', 'a desert vegetation', 'a diner indoor', 'a diner outdoor', 'a dinette home', 'a dinette vehicle', 'a dining car', 'a dining room', 'a discotheque', 'a dock', 'a doorway outdoor', 'a dorm room', 'a driveway', 'a driving range outdoor', 'a drugstore', 'a electrical substation', 'a elevator door', 'a elevator interior', 'a elevator shaft', 'a engine room', 'a escalator indoor', 'a excavation', 'a factory indoor', 'a fairway', 'a fastfood restaurant', 'a field cultivated', 'a field wild', 'a fire escape', 'a fire station', 'a firing range indoor', 'a fishpond', 'a florist shop indoor', 'a food court', 'a forest broadleaf', 'a forest needleleaf', 'a forest path', 'a forest road', 'a formal garden', 'a fountain', 'a galley', 'a game room', 'a garage indoor', 'a garbage dump', 'a gas station', 'a gazebo exterior', 'a general store indoor', 'a general store outdoor', 'a gift shop', 'a golf course', 'a greenhouse indoor', 'a greenhouse outdoor', 'a gymnasium indoor', 'a hangar indoor', 'a hangar outdoor', 'a harbor', 'a hayfield', 'a heliport', 'a herb garden', 'a highway', 'a hill', 'a home office', 'a hospital', 'a hospital room', 'a hot spring', 'a hot tub outdoor', 'a hotel outdoor', 'a hotel room', 'a house', 'a hunting lodge outdoor', 'a ice cream parlor', 'a ice floe', 'a ice shelf', 'a ice skating rink indoor', 'a ice skating rink outdoor', 'a iceberg', 'a igloo', 'a industrial area', 'a inn outdoor', 'a islet', 'a jacuzzi indoor', 'a jail cell', 'a jail indoor', 'a jewelry shop', 'a kasbah', 'a kennel indoor', 'a kennel outdoor', 'a kindergarden classroom', 'a kitchen', 'a kitchenette', 'a labyrinth outdoor', 'a lake natural', 'a landfill', 'a landing deck', 'a laundromat', 'a lecture room', 'a library indoor', 'a library outdoor', 'a lido deck outdoor', 'a lift bridge', 'a lighthouse', 'a limousine interior', 'a living room', 'a lobby', 'a lock chamber', 'a locker room', 'a mansion', 'a manufactured home', 'a market indoor', 'a market outdoor', 'a marsh', 'a martial arts gym', 'a mausoleum', 'a medina', 'a moat water', 'a monastery outdoor', 'a mosque indoor', 'a mosque outdoor', 'a motel', 'a mountain', 'a mountain snowy', 'a movie theater indoor', 'a museum indoor', 'a music store', 'a music studio', 'a nuclear power plant outdoor', 'a nursery', 'a oast house', 'a observatory outdoor', 'a ocean', 'a office', 'a office building', 'a oil refinery outdoor', 'a oilrig', 'a operating room', 'a orchard', 'a outhouse outdoor', 'a pagoda', 'a palace', 'a pantry', 'a park', 'a parking garage indoor', 'a parking garage outdoor', 'a parking lot', 'a parlor', 'a pasture', 'a patio', 'a pavilion', 'a pharmacy', 'a phone booth', 'a physics laboratory', 'a picnic area', 'a pilothouse indoor', 'a planetarium outdoor', 'a playground', 'a playroom', 'a plaza', 'a podium indoor', 'a podium outdoor', 'a pond', 'a poolroom establishment', 'a poolroom home', 'a power plant outdoor', 'a promenade deck', 'a pub indoor', 'a pulpit', 'a putting green', 'a racecourse', 'a raceway', 'a raft', 'a railroad track', 'a rainforest', 'a reception', 'a recreation room', 'a residential neighborhood', 'a restaurant', 'a restaurant kitchen', 'a restaurant patio', 'a rice paddy', 'a riding arena', 'a river', 'a rock arch', 'a rope bridge', 'a ruin', 'a runway', 'a sandbar', 'a sandbox', 'a sauna', 'a schoolhouse', 'a sea cliff', 'a server room', 'a shed', 'a shoe shop', 'a shopfront', 'a shopping mall indoor', 'a shower', 'a skatepark', 'a ski lodge', 'a ski resort', 'a ski slope', 'a sky', 'a skyscraper', 'a slum', 'a snowfield', 'a squash court', 'a stable', 'a stadium baseball', 'a stadium football', 'a stage indoor', 'a staircase', 'a street', 'a subway interior', 'a subway station platform', 'a supermarket', 'a sushi bar', 'a swamp', 'a swimming pool indoor', 'a swimming pool outdoor', 'a synagogue indoor', 'a synagogue outdoor', 'a television studio', 'a temple east asia', 'a temple south asia', 'a tennis court indoor', 'a tennis court outdoor', 'a tent outdoor', 'a theater indoor procenium', 'a theater indoor seats', 'a thriftshop', 'a throne room', 'a ticket booth', 'a toll plaza', 'a topiary garden', 'a tower', 'a toyshop', 'a track outdoor', 'a train railway', 'a train station platform', 'a tree farm', 'a tree house', 'a trench', 'a underwater coral reef', 'a utility room', 'a valley', 'a van interior', 'a vegetable garden', 'a veranda', 'a veterinarians office', 'a viaduct', 'a videostore', 'a village', 'a vineyard', 'a volcano', 'a volleyball court indoor', 'a volleyball court outdoor', 'a waiting room', 'a warehouse indoor', 'a water tower', 'a waterfall block', 'a waterfall fan', 'a waterfall plunge', 'a watering hole', 'a wave', 'a wet bar', 'a wheat field', 'a wind farm', 'a windmill', 'a wine cellar barrel storage', 'a wine cellar bottle storage', 'a wrestling ring indoor', 'a yard', 'a youth hostel']

SUN397_names_coop = \
['abbey', 'airplane cabin', 'aquarium', 'closet', 'clothing store', 'coast', 'cockpit', 'coffee shop', 'computer room', 'conference center', 'conference room', 'construction site', 'control room', 'aqueduct', 'outdoor control tower', 'corn field', 'corral', 'corridor', 'cottage garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered bridge', 'creek', 'arch', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'archive', 'home dinette', 'vehicle dinette', 'dining car', 'dining room', 'discotheque', 'dock', 'outdoor doorway', 'dorm room', 'driveway', 'outdoor driving range', 'outdoor arrival gate', 'drugstore', 'electrical substation', 'door elevator', 'interior elevator', 'elevator shaft', 'engine room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'art gallery', 'fastfood restaurant', 'cultivated field', 'wild field', 'fire escape', 'fire station', 'indoor firing range', 'fishpond', 'indoor florist shop', 'food court', 'broadleaf forest', 'art school', 'needleleaf forest', 'forest path', 'forest road', 'formal garden', 'fountain', 'galley', 'game room', 'indoor garage', 'garbage dump', 'gas station', 'art studio', 'exterior gazebo', 'indoor general store', 'outdoor general store', 'gift shop', 'golf course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'assembly line', 'harbor', 'hayfield', 'heliport', 'herb garden', 'highway', 'hill', 'home office', 'hospital', 'hospital room', 'hot spring', 'outdoor athletic field', 'outdoor hot tub', 'outdoor hotel', 'hotel room', 'house', 'outdoor hunting lodge', 'ice cream parlor', 'ice floe', 'ice shelf', 'indoor ice skating rink', 'outdoor ice skating rink', 'airport terminal', 'public atrium', 'iceberg', 'igloo', 'industrial area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail cell', 'jewelry shop', 'kasbah', 'attic', 'indoor kennel', 'outdoor kennel', 'kindergarden classroom', 'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing deck', 'laundromat', 'auditorium', 'lecture room', 'indoor library', 'outdoor library', 'outdoor lido deck', 'lift bridge', 'lighthouse', 'limousine interior', 'living room', 'lobby', 'lock chamber', 'auto factory', 'locker room', 'mansion', 'manufactured home', 'indoor market', 'outdoor market', 'marsh', 'martial arts gym', 'mausoleum', 'medina', 'water moat', 'badlands', 'outdoor monastery', 'indoor mosque', 'outdoor mosque', 'motel', 'mountain', 'mountain snowy', 'indoor movie theater', 'indoor museum', 'music store', 'music studio', 'indoor badminton court', 'outdoor nuclear power plant', 'nursery', 'oast house', 'outdoor observatory', 'ocean', 'office', 'office building', 'outdoor oil refinery', 'oilrig', 'operating room', 'baggage claim', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry', 'park', 'indoor parking garage', 'outdoor parking garage', 'parking lot', 'parlor', 'shop bakery', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone booth', 'physics laboratory', 'picnic area', 'indoor pilothouse', 'outdoor planetarium', 'playground', 'exterior balcony', 'playroom', 'plaza', 'indoor podium', 'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power plant', 'promenade deck', 'indoor pub', 'interior balcony', 'pulpit', 'putting green', 'racecourse', 'raceway', 'raft', 'railroad track', 'rainforest', 'reception', 'recreation room', 'residential neighborhood', 'alley', 'ball pit', 'restaurant', 'restaurant kitchen', 'restaurant patio', 'rice paddy', 'riding arena', 'river', 'rock arch', 'rope bridge', 'ruin', 'runway', 'ballroom', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea cliff', 'server room', 'shed', 'shoe shop', 'shopfront', 'indoor shopping mall', 'bamboo forest', 'shower', 'skatepark', 'ski lodge', 'ski resort', 'ski slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash court', 'banquet hall', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase', 'street', 'subway interior', 'platform subway station', 'supermarket', 'sushi bar', 'bar', 'swamp', 'indoor swimming pool', 'outdoor swimming pool', 'indoor synagogue', 'outdoor synagogue', 'television studio', 'east asia temple', 'south asia temple', 'indoor tennis court', 'outdoor tennis court', 'barn', 'outdoor tent', 'indoor procenium theater', 'indoor seats theater', 'thriftshop', 'throne room', 'ticket booth', 'toll plaza', 'topiary garden', 'tower', 'toyshop', 'barndoor', 'outdoor track', 'train railway', 'platform train station', 'tree farm', 'tree house', 'trench', 'coral reef underwater', 'utility room', 'valley', 'van interior', 'baseball field', 'vegetable garden', 'veranda', 'veterinarians office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'indoor volleyball court', 'outdoor volleyball court', 'basement', 'waiting room', 'indoor warehouse', 'water tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering hole', 'wave', 'wet bar', 'wheat field', 'basilica', 'wind farm', 'windmill', 'barrel storage wine cellar', 'bottle storage wine cellar', 'indoor wrestling ring', 'yard', 'youth hostel', 'amphitheater', 'outdoor basketball court', 'bathroom', 'batters box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty salon', 'bedroom', 'berth', 'amusement arcade', 'biology laboratory', 'indoor bistro', 'boardwalk', 'boat deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical garden', 'indoor bow window', 'outdoor bow window', 'amusement park', 'bowling alley', 'boxing ring', 'indoor brewery', 'bridge', 'building facade', 'bullring', 'burial chamber', 'bus interior', 'butchers shop', 'butte', 'anechoic chamber', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy store', 'canyon', 'backseat car interior', 'frontseat car interior', 'outdoor apartment building', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese factory', 'indoor apse', 'chemistry lab', 'indoor chicken coop', 'outdoor chicken coop', 'childs room', 'indoor church', 'outdoor church', 'classroom', 'clean room', 'cliff', 'indoor cloister']


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


class TransferDS:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = os.path.join(location, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = classnames

class Aircraft_old:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = os.path.join(location, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)

        self.test_dataset = aircraft.FGVCAircraft(root='/opt/tiger/filter_transfer/data/fgvc-aircraft-2013b', train=False,
                                transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = [v[2:] for v in aircraft_names]

class Aircraft:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/aircraft_train_val/val'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = aircraft_names

class Birds:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = os.path.join(location, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in birds_names]
        self.classnames = [v for v in birds_names]

class Cars:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/cars_train_val_test/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in cars_names]
        self.classnames = [v for v in cars_names_coop]

def sampleFromClass(ds, k):
    class_counts = {}
    indices = []

    for i in range(len(ds)):
        c = ds[i][1]
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            indices.append(i)
        if len(indices) == k * 10:
            return indices

    return indices

def sampleFromClass_random(ds, k, random_seed=0):
    class_idxs = {}
    indices = []
    random.seed(random_seed)

    for j in range(100):
        class_idxs[j] = []

    for i in range(len(ds)):
        c = ds[i][1]
        class_idxs[c].append(i)

    for j in range(100):
        indices.extend(random.sample(class_idxs[j], k))

    return indices


class Cifar10:
    'Cifar10 png version, using https://github.com/knjcode/cifar2png'
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/cifar_png/cifar10png/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in cars_names]
        self.classnames = cifar10_classnames

class Cifar100:
    'Cifar100 png version, using https://github.com/knjcode/cifar2png'

    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/cifar_png/cifar100png/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in cars_names]
        self.classnames = cifar100_names


class CALtech101:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        ds = Caltech101('/opt/tiger/filter_transfer/data/', download=True)
        np.random.seed(0)
        ch.manual_seed(0)
        ch.cuda.manual_seed(0)
        ch.cuda.manual_seed_all(0)
        NUM_TRAINING_SAMPLES_PER_CLASS = 30

        class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]

        train_indices = sum([np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in
                             class_start_idx], [])
        test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))

        train_set = Subset(ds, train_indices)
        test_set = Subset(ds, test_indices)

        train_set = TransformedDataset(train_set, transform=preprocess)
        test_set = TransformedDataset(test_set, transform=preprocess)

        if location == '/opt/tiger/filter_transfer/data/':
            self.train_dataset = train_set
            print('training on caltech101 original real data')
        else:
            print('training on caltech101 glide')
            train_path = os.path.join(location, 'train')
            if not os.path.exists(train_path):
                raise ValueError("Train data must be stored in {0}".format(train_path))
            self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)

        self.train_loader = ch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = test_set

        self.test_loader = ch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = [v for v in caltech101_clip_names]

class CALtech101_coop:
    '''
    Caltech 101 Coop split: only 100 classes
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/caltech101_train_val_test/test'
        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)

        self.train_loader = ch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.test_loader = ch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = [v[2:] for v in caltech101_coop_names]


class CALtech256:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        ds = Caltech256('/opt/tiger/filter_transfer/data/', download=True)
        np.random.seed(0)
        ch.manual_seed(0)
        ch.cuda.manual_seed(0)
        ch.cuda.manual_seed_all(0)
        NUM_TRAINING_SAMPLES_PER_CLASS = 60

        class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]

        train_indices = sum([np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in
                             class_start_idx], [])
        test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))

        train_set = Subset(ds, train_indices)
        test_set = Subset(ds, test_indices)

        train_set = TransformedDataset(train_set, transform=preprocess)
        test_set = TransformedDataset(test_set, transform=preprocess)

        if location == '/opt/tiger/filter_transfer/data/':
            self.train_dataset = train_set
            print('training on caltech256 original real data')
        else:
            print('training on caltech256_glide')
            train_path = os.path.join(location, 'train')
            if not os.path.exists(train_path):
                raise ValueError("Train data must be stored in {0}".format(train_path))
            self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)


        self.train_loader = ch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = test_set

        self.test_loader = ch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = [v[2:] for v in caltech256_names]

class Cub:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/CUB_200_2011/val'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = [v[2:] for v in cub_names]

class Dtd:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):


        train_path = os.path.join(location, 'train')
        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        print('training on dtd_glide')

        self.train_loader = ch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_path = '/opt/tiger/filter_transfer/data/dtd_train_val_test/test'
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.test_loader = ch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = [v[2:] for v in dtd_names]

class Eurosat:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/EuroSAT_train_val_test/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in eurosat_names]
        self.classnames = [v[2:] for v in eurosat_names_coop]

class Flowers:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = os.path.join(location, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = [v[2:] for v in flowers_names]

class Food:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        food = food_101.FOOD101(transform=preprocess)
        train_set, test_set, classes = food.get_dataset()

        if location == '/opt/tiger/filter_transfer/data/':
            self.train_dataset = train_set
            print('training on food original real data')
        else:
            print('training on food_glide')
            train_path = os.path.join(location, 'train')
            if not os.path.exists(train_path):
                raise ValueError("Train data must be stored in {0}".format(train_path))
            self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)

        self.train_loader = ch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = test_set

        self.test_loader = ch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # self.classnames = [v[2:] for v in food_names]
        self.classnames = [v for v in food_names]

class Pets:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/pets_train_val_test/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = [v for v in pets_names]

class Sun397:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        # test_path = os.path.join(location, 'val')
        test_path = '/opt/tiger/filter_transfer/data/SUN397_train_val_test/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = [v for v in SUN397_names_coop]

class Imgnet_glide:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = location
        test_path = '/opt/tiger/filter_transfer/data/imagenet/ILSVRC2012_img_train_128k/val'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = openai_classnames

class Imgnet_r:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/imagenet-r'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classnames = name_imagenet_r



class ImageNetTransfer(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'custom_class': None,
            'std': ch.tensor(kwargs['std']),
            'transform_train': cs.TRAIN_TRANSFORMS,
            'label_mapping': None,
            'transform_test': cs.TEST_TRANSFORMS
        }
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)

class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3,1,1)
        return sample, label

def make_loaders_pets(batch_size, workers):
    # ds = ImageNetTransfer(cs.PETS_PATH, num_classes=37, name='pets',mean=[0., 0., 0.], std=[1., 1., 1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/pets', num_classes=37, name='pets',mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_pets_v(batch_size, workers):
    # ds = ImageNetTransfer(cs.PETS_PATH, num_classes=37, name='pets',mean=[0., 0., 0.], std=[1., 1., 1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/pets_v3_20k_sc', num_classes=37, name='pets',mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_birds(batch_size, workers):
    # ds = ImageNetTransfer(cs.BIRDS_PATH, num_classes=500, name='birds',mean=[0.,0.,0.], std=[1.,1.,1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/birdsnap', num_classes=500, name='birds',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_birds_v(batch_size, workers):
    # ds = ImageNetTransfer(cs.BIRDS_PATH, num_classes=500, name='birds',mean=[0.,0.,0.], std=[1.,1.,1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/birds_v1_20k_sc', num_classes=500, name='birds',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_imgnet(batch_size, workers):
    ds = ImageNet(cs.IMGNET_PATH)
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_imgnet_clip(batch_size, workers):
    ds = ImageNet(cs.IMGNET_PATH, mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)


def make_loaders_cub(batch_size, workers):
    ds = ImageNetTransfer(cs.CUB_PATH, num_classes=200, name='cub',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_cub_v(batch_size, workers):
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/cub_v1_20k_sc/', num_classes=200, name='cub_v',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_coco(batch_size, workers):
    ds = ImageNetTransfer(cs.COCO_PATH, num_classes=1, name='coco',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_ade20k(batch_size, workers):
    ds = ImageNetTransfer(cs.ADE20K_PATH, num_classes=1, name='ade20k',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_SUN(batch_size, workers):
    # ds = ImageNetTransfer(cs.SUN_PATH, num_classes=397, name='SUN397',mean=[0.,0.,0.], std=[1.,1.,1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/SUN397/splits_01', num_classes=397, name='SUN397',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_SUN_v(batch_size, workers):
    # ds = ImageNetTransfer(cs.SUN_PATH, num_classes=397, name='SUN397',mean=[0.,0.,0.], std=[1.,1.,1.])
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/sun_v3_20k_sc', num_classes=397, name='SUN397',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_CIFAR10(batch_size, workers, subset):
    ds = CIFAR('/tmp')
    ds.transform_train = cs.TRAIN_TRANSFORMS
    ds.transform_test = cs.TEST_TRANSFORMS
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_CIFAR10_v(batch_size, workers, subset):
    ds = CIFAR('/tmp')
    ds.transform_train = cs.TRAIN_TRANSFORMS
    ds.transform_test = cs.TEST_TRANSFORMS

    train_loader_abd, validation_loader = ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/cifar10_v1_20k_sc/', num_classes=10, name='cifar10_v',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

    return ds, (train_loader, validation_loader)

def make_loaders_CIFAR100(batch_size, workers, subset):
    ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_CIFAR100_v(batch_size, workers, subset):
    ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    train_loader_abd, validation_loader = ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/cifar100_v1_20k_sc/', num_classes=100, name='cifar100_v',mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers, subset=subset)
    return ds, (train_loader, validation_loader)

def make_loaders_oxford(batch_size, workers):
    # ds = ImageNetTransfer(cs.FLOWERS_PATH, num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    # ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/flowers_new/', num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    ds = ImageNet('/opt/tiger/filter_transfer/data/flowers_new/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_oxford_v(batch_size, workers):
    # ds = ImageNetTransfer(cs.FLOWERS_PATH, num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    # ds = ImageNet('/opt/tiger/filter_transfer/data/flowers_v1/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/flowers_v1_20k_sc/', num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_glide(batch_size, workers):
    # ds = ImageNetTransfer(cs.FLOWERS_PATH, num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    # ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/flowers_new/', num_classes=102,name='oxford_flowers', mean=[0.,0.,0.],std=[1.,1.,1.])
    # ds = ImageNet('/opt/tiger/filter_transfer/data/pool15/birds_v1_20k/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNet('/opt/tiger/filter_transfer/data/sun_v3_20k/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_glide2(batch_size, workers):
   # ds = ImageNet('/opt/tiger/filter_transfer/data/pool15/birds_v1_20k/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNet('/opt/tiger/filter_transfer/data/cifar10_v1_20k/', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]),std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

# def make_loaders_aircraft(batch_size, workers):
#     # ds = ImageNetTransfer(cs.FGVC_PATH, num_classes=100, name='aircraft',mean=[0.,0.,0.], std=[1.,1.,1.])
#     ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/fgvc-aircraft-2013b', num_classes=100, name='aircraft',mean=[0.,0.,0.], std=[1.,1.,1.])
#     ds.custom_class = aircraft.FGVCAircraft
#     return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_aircraft(batch_size, workers):
    ds = ImageNet('/opt/tiger/filter_transfer/data/fgvc-aircraft-2013b', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds.custom_class = aircraft.FGVCAircraft
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_aircraft_v(batch_size, workers):
    # ds = ImageNet('/opt/tiger/filter_transfer/data/aircraft_v1', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/fgvc-aircraft-2013b', num_classes=100, name='aircraft', mean=[0., 0., 0.], std=[1., 1., 1.])
    ds.custom_class = aircraft.FGVCAircraft

    train_loader_abd, validation_loader = ds.make_loaders(batch_size=batch_size, workers=workers)

    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/aircraft_v1_20k_sc/', num_classes=100, name='aircraft_v1_20k_sc',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers)

    return ds, (train_loader, validation_loader)

def make_loaders_food(batch_size, workers):
    food = food_101.FOOD101()
    train_ds, valid_ds, classes =  food.get_dataset()
    train_dl, valid_dl = food.get_dls(train_ds, valid_ds, bs=batch_size,
                                                    num_workers=workers)
    return 101, (train_dl, valid_dl)

def make_loaders_food_v(batch_size, workers):
    food = food_101.FOOD101()
    train_ds, valid_ds, classes =  food.get_dataset()
    train_loader_abd, validation_loader = food.get_dls(train_ds, valid_ds, bs=batch_size,
                                                    num_workers=workers)
    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/food_v3_20k_sc/', num_classes=101,
                            name='food_v3_20k_sc',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers)
    return 101, (train_loader, validation_loader)
    # return 101, (train_dl, valid_dl)

def make_loaders_caltech101_v(batch_size, workers):
    ds = Caltech101(cs.CALTECH101_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    train_loader_abd, validation_loader = [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]

    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/caltech101_v1_20k_sc/', num_classes=101,
                            name='caltech101_v1_20k_sc',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers)

    return 101, (train_loader, validation_loader)
    # return 101, [DataLoader(d, batch_size=batch_size, shuffle=True,
    #             num_workers=workers) for d in (train_set, test_set)]

def make_loaders_caltech101(batch_size, workers):
    ds = Caltech101(cs.CALTECH101_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 101, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]

def make_loaders_caltech256(batch_size, workers):
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]

def make_loaders_caltech256_v(batch_size, workers):
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    train_loader_abd, validation_loader = [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]
    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/caltech256_v1_20k_sc/', num_classes=257,
                            name='caltech256_v1_20k_sc',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers)

    return 257, (train_loader, validation_loader)
    # return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
    #             num_workers=workers) for d in (train_set, test_set)]

def make_loaders_dtd(batch_size, workers):
        train_set = dtd.DTD(train=True)
        val_set = dtd.DTD(train=False)
        return 47, [DataLoader(ds, batch_size=batch_size, shuffle=True,
                num_workers=workers) for ds in (train_set, val_set)]


def make_loaders_dtd_v(batch_size, workers):
    train_set = dtd.DTD(train=True)
    val_set = dtd.DTD(train=False)

    train_loader_abd, validation_loader = [DataLoader(ds, batch_size=batch_size, shuffle=True,
                           num_workers=workers) for ds in (train_set, val_set)]

    ds_v = ImageNetTransfer('/opt/tiger/filter_transfer/data/dtd_v3_20k_sc/', num_classes=47,
                            name='dtd_v3_20k_sc',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    train_loader, validation_loader_abd = ds_v.make_loaders(batch_size=batch_size, workers=workers)
    return 47, (train_loader, validation_loader)
    # return 47, [DataLoader(ds, batch_size=batch_size, shuffle=True,
    #                        num_workers=workers) for ds in (train_set, val_set)]

# def make_loaders_cars(batch_size, workers):
#     # ds = ImageNetTransfer(cs.CARS_PATH, num_classes=196, name='stanford_cars',mean=[0.,0.,0.], std=[1.,1.,1.])
#     ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/cars_new', num_classes=196, name='stanford_cars',mean=[0.,0.,0.], std=[1.,1.,1.])
#     return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_cars(batch_size, workers):
    # ds = ImageNet('/opt/tiger/filter_transfer/data/cars_new', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/cars_new', num_classes=196, name='cars', mean=[0., 0., 0.],std=[1., 1., 1.])
    # ds = ImageNet('/opt/tiger/filter_transfer/data/cars_v2', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_cars_v(batch_size, workers):
    # ds = ImageNet('/opt/tiger/filter_transfer/data/cars_new', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNet('/opt/tiger/filter_transfer/data/cars_v1_20k_sc', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    # ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/cars_v1_20k_sc', num_classes=196, name='cars_v', mean=[0., 0., 0.],std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_cars_vt(batch_size, workers):
    # ds = ImageNet('/opt/tiger/filter_transfer/data/cars_new', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    # ds = ImageNet('/opt/tiger/filter_transfer/data/cars_v1_20k_sc', mean=ch.tensor([0.48145466, 0.4578275, 0.40821073]), std=ch.tensor([0.26862954, 0.26130258, 0.27577711]))
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/cars_v1_20k_sc', num_classes=196, name='cars_v', mean=[0., 0., 0.],std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_eurosat(batch_size, workers):
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/EuroSAT', num_classes=10, name='eurosat',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_eurosat_v(batch_size, workers):
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/eurosat_v3_20k_sc', num_classes=10, name='eurosat',mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_ucf101(batch_size, workers):
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/UCF-101', num_classes=101, name='ucf101',mean=[0.,0.,0.], std=[1.,1.,1.])
    # ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/UCF-101', num_classes=101, name='ucf101',mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_ucf101_v(batch_size, workers):
    ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/ucf101_v1_20k_sc', num_classes=101, name='ucf101_v1_20k_sc',mean=[0.,0.,0.], std=[1.,1.,1.])
    # ds = ImageNetTransfer('/opt/tiger/filter_transfer/data/UCF-101', num_classes=101, name='ucf101',mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_mix4(batch_size, workers):
    # ---- dtd
    dtd_train_set = dtd.DTD(train=True)
    dtd_train_loader = DataLoader(dtd_train_set, batch_size=batch_size, shuffle=True,
                num_workers=workers)

    # ---- caltech256
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    caltech256_train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    caltech256_test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    caltech256_train_loader = DataLoader(caltech256_train_set, batch_size=batch_size, shuffle=True,
               num_workers=workers)

    # ---- cub
    cub_ds = ImageNetTransfer(cs.CUB_PATH, num_classes=200, name='cub',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    cub_train_loader, cub_test_loader = cub_ds.make_loaders(batch_size=batch_size, workers=workers)

    # ---- cifar100
    cifar100_ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                          mean=[0.5071, 0.4867, 0.4408],
                          std=[0.2675, 0.2565, 0.2761])
    cifar100_ds.custom_class = CIFAR100
    cifar100_train_loader, _ = cifar100_ds.make_loaders(batch_size=batch_size, workers=workers)

    return 200, [dtd_train_loader, caltech256_train_loader, cub_train_loader, cifar100_train_loader], cub_test_loader

def make_loaders_mix_seg(batch_size, workers):
    ds = ImageNetTransfer(cs.MIX_SEG_PATH, num_classes=1000, name='mix_seg',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

DS_TO_FUNC = {
    "dtd": make_loaders_dtd,
    "dtd_v": make_loaders_dtd_v,
    "cars": make_loaders_cars,
    "cars_v": make_loaders_cars_v,
    "cars_vt": make_loaders_cars_vt,
    "cifar10": make_loaders_CIFAR10,
    "cifar10_v": make_loaders_CIFAR10_v,
    "cifar100": make_loaders_CIFAR100,
    "cifar100_v": make_loaders_CIFAR100_v,
    "SUN397": make_loaders_SUN,
    "SUN397_v": make_loaders_SUN_v,
    "aircraft": make_loaders_aircraft,
    "aircraft_v": make_loaders_aircraft_v,
    "flowers": make_loaders_oxford,
    "flowers_v": make_loaders_oxford_v,
    "food": make_loaders_food,
    "food_v": make_loaders_food_v,
    "birds": make_loaders_birds,
    "birds_v": make_loaders_birds_v,
    "cub": make_loaders_cub,
    "cub_v": make_loaders_cub_v,
    "coco": make_loaders_coco,
    "ade20k": make_loaders_ade20k,
    "caltech101": make_loaders_caltech101,
    "caltech101_v": make_loaders_caltech101_v,
    "caltech256": make_loaders_caltech256,
    "caltech256_v": make_loaders_caltech256_v,
    "pets": make_loaders_pets,
    "pets_v": make_loaders_pets_v,
    "eurosat": make_loaders_eurosat,
    "eurosat_v": make_loaders_eurosat_v,
    "ucf101": make_loaders_ucf101,
    "ucf101_v": make_loaders_ucf101_v,
    "imgnet": make_loaders_imgnet,
    "imgnet_clip": make_loaders_imgnet_clip,
    'mix4': make_loaders_mix4,
    'mix_seg': make_loaders_mix_seg,
    'glide': make_loaders_glide,
    'glide2': make_loaders_glide2,
}

def make_loaders(ds, batch_size, workers, subset):
    if ds in ['cifar10', 'cifar100', 'cifar100_v', 'cifar10_v']:
        return DS_TO_FUNC[ds](batch_size, workers, subset)

    if subset: raise Exception(f'Subset not supported for the {ds} dataset')
    return DS_TO_FUNC[ds](batch_size, workers)

if __name__ == "__main__":
    # make_loaders_mix4(16, 1)
    make_loaders_food(16, 1)