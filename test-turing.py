from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 2
main(options)
