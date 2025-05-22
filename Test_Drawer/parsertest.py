import argparse

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
 
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
 
# Add an argument for the list of integers
parser.add_argument('--hidden', type=list_of_ints)
 
# Parse the command-line arguments
args = parser.parse_args()
 
# Use the list of integers in your script
print(args.int_list)