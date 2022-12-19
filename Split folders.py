import splitfolders as sf

input_folder = 'flowers'
output_folder = '/home/oscar6/Research/Resnet-50/processed_stuff'
#This is to split the data into the correct proportions needed for the data.
#The ratio controls the amount of split, ratio(training, testing, validations)
sf.ratio(input_folder, output_folder, seed=42, ratio=(.6, .2, .2))